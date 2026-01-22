#!/usr/bin/env python3
"""
Distributed Testing Framework - Test Runner

This module provides a command-line interface for running tests with the distributed
testing framework. It can run in different modes:

1. Coordinator mode: Start a coordinator server that distributes tasks
2. Worker mode: Start worker nodes that execute tasks
3. Client mode: Submit tasks to a running coordinator
4. Dashboard mode: Start a dashboard server for monitoring
5. All mode: Start a coordinator, workers, and dashboard for testing

Usage:
    # Run in coordinator mode
    python run_test.py --mode coordinator --host 0.0.0.0 --port 8080
    
    # Run in worker mode
    python run_test.py --mode worker --coordinator http://localhost:8080 --api-key KEY
    
    # Run in client mode (submit tasks)
    python run_test.py --mode client --coordinator http://localhost:8080 --test-file test_file.py
    
    # Run in dashboard mode
    python run_test.py --mode dashboard --coordinator http://localhost:8080
    
    # Run all components (for testing)
    python run_test.py --mode all --host localhost
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import subprocess
import threading
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("run_test")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import from distributed_testing package
try:
    from duckdb_api.distributed_testing.coordinator import CoordinatorServer
    from duckdb_api.distributed_testing.worker import WorkerClient
    from duckdb_api.distributed_testing.dashboard_server import DashboardServer
    DIRECT_IMPORT = True
except ImportError:
    logger.warning("Could not import directly from distributed_testing modules, will use subprocess")
    DIRECT_IMPORT = False

# Test modes
MODE_COORDINATOR = "coordinator"
MODE_WORKER = "worker"
MODE_CLIENT = "client"
MODE_DASHBOARD = "dashboard"
MODE_ALL = "all"

# Default values
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DEFAULT_DASHBOARD_PORT = 8081
DEFAULT_DB_PATH = None  # Will use in-memory database if None
DEFAULT_WORKER_COUNT = 2
DEFAULT_TEST_TIMEOUT = 600  # 10 minutes
DEFAULT_SECURITY_CONFIG = "security_config.json"


def run_coordinator(host: str, port: int, db_path: Optional[str] = None,
                   security_config: Optional[str] = None) -> subprocess.Popen:
    """Run the coordinator server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        db_path: Optional path to DuckDB database
        security_config: Optional path to security configuration file
        
    Returns:
        Subprocess object if using subprocess, None if using direct import
    """
    if DIRECT_IMPORT:
        # Create and start coordinator in a thread
        def coordinator_thread():
            # Create coordinator
            coordinator = CoordinatorServer(
                host=host,
                port=port,
                db_path=db_path,
                token_secret=None  # Will be auto-generated
            )
            
            # Start coordinator
            try:
                logger.info(f"Starting coordinator on {host}:{port}...")
                coordinator.start()
            except KeyboardInterrupt:
                logger.info("Coordinator interrupted by user")
            
        thread = threading.Thread(target=coordinator_thread, daemon=True)
        thread.start()
        
        logger.info(f"Started coordinator server on {host}:{port}")
        return None
    else:
        # Build command
        cmd = [sys.executable, "-m", "duckdb_api.distributed_testing.coordinator"]
        cmd.extend(["--host", host])
        cmd.extend(["--port", str(port)])
        
        if db_path:
            cmd.extend(["--db-path", db_path])
            
        if security_config:
            cmd.extend(["--security-config", security_config])
        
        # Start process
        logger.info(f"Starting coordinator process: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait a bit for startup
        time.sleep(2)
        
        return process


def run_worker(coordinator_url: str, api_key: str, worker_id: Optional[str] = None,
              work_dir: Optional[str] = None) -> subprocess.Popen:
    """Run a worker node.
    
    Args:
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        worker_id: Optional worker ID (generated if not provided)
        work_dir: Optional working directory for tasks
        
    Returns:
        Subprocess object if using subprocess, None if using direct import
    """
    if DIRECT_IMPORT:
        # Create and start worker in a thread
        def worker_thread():
            # Create worker
            worker = WorkerClient(
                coordinator_url=coordinator_url,
                api_key=api_key,
                worker_id=worker_id
            )
            
            # Start worker
            try:
                logger.info(f"Starting worker {worker.worker_id}...")
                worker.run()
            except KeyboardInterrupt:
                logger.info("Worker interrupted by user")
            
        thread = threading.Thread(target=worker_thread, daemon=True)
        thread.start()
        
        logger.info(f"Started worker connecting to {coordinator_url}")
        return None
    else:
        # Build command
        cmd = [sys.executable, "-m", "duckdb_api.distributed_testing.worker"]
        cmd.extend(["--coordinator", coordinator_url])
        cmd.extend(["--api-key", api_key])
        
        if worker_id:
            cmd.extend(["--worker-id", worker_id])
            
        if work_dir:
            cmd.extend(["--work-dir", work_dir])
        
        # Start process
        logger.info(f"Starting worker process: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait a bit for startup
        time.sleep(1)
        
        return process


def run_dashboard(host: str, port: int, coordinator_url: str,
                auto_open: bool = False) -> subprocess.Popen:
    """Run the dashboard server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        coordinator_url: URL of the coordinator server
        auto_open: Whether to automatically open the dashboard in a browser
        
    Returns:
        Subprocess object if using subprocess, None if using direct import
    """
    if DIRECT_IMPORT:
        # Create and start dashboard in a thread
        def dashboard_thread():
            # Create dashboard
            dashboard = DashboardServer(
                host=host,
                port=port,
                coordinator_url=coordinator_url,
                auto_open=auto_open
            )
            
            # Start dashboard
            try:
                logger.info(f"Starting dashboard on {host}:{port}...")
                dashboard.start()
                
                # Keep thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Dashboard interrupted by user")
                dashboard.stop()
            
        thread = threading.Thread(target=dashboard_thread, daemon=True)
        thread.start()
        
        logger.info(f"Started dashboard server on http://{host}:{port}")
        return None
    else:
        # Build command
        cmd = [sys.executable, "-m", "duckdb_api.distributed_testing.dashboard_server"]
        cmd.extend(["--host", host])
        cmd.extend(["--port", str(port)])
        cmd.extend(["--coordinator-url", coordinator_url])
        
        if auto_open:
            cmd.append("--auto-open")
        
        # Start process
        logger.info(f"Starting dashboard process: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait a bit for startup
        time.sleep(1)
        
        return process


def submit_test_task(coordinator_url: str, test_file: str, test_args: List[str] = None,
                    priority: int = 5) -> str:
    """Submit a test task to the coordinator.
    
    Args:
        coordinator_url: URL of the coordinator server
        test_file: Path to the test file
        test_args: Optional list of arguments for the test
        priority: Priority of the task (lower is higher priority)
        
    Returns:
        Task ID if successful, None otherwise
    """
    import requests
    
    try:
        # Prepare task data
        task_data = {
            "type": "test",
            "priority": priority,
            "config": {
                "test_file": test_file,
                "test_args": test_args or []
            },
            "requirements": {}
        }
        
        # Determine if test file has specific hardware requirements
        if os.path.exists(test_file):
            # Look for hardware-related content in the file
            with open(test_file, "r") as f:
                content = f.read()
                
                # Check for hardware requirements
                if "cuda" in content.lower() or "gpu" in content.lower():
                    task_data["requirements"]["hardware"] = ["cuda"]
                    
                if "webgpu" in content.lower():
                    task_data["requirements"]["hardware"] = ["webgpu"]
                    
                if "webnn" in content.lower():
                    task_data["requirements"]["hardware"] = ["webnn"]
                    
                # Check for memory requirements
                if "memory_gb" in content.lower():
                    # Simple heuristic - more stringent requirements can be added later
                    task_data["requirements"]["min_memory_gb"] = 4
        
        # Submit task
        api_url = f"{coordinator_url.rstrip('/')}/api/tasks"
        response = requests.post(api_url, json=task_data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                task_id = result.get("task_id")
                logger.info(f"Submitted test task {task_id} for {test_file}")
                return task_id
            else:
                logger.error(f"Error submitting task: {result.get('error')}")
        else:
            logger.error(f"Error submitting task: HTTP {response.status_code}")
            
        return None
    except Exception as e:
        logger.error(f"Error submitting test task: {e}")
        return None


def wait_for_task_completion(coordinator_url: str, task_id: str, 
                           timeout: int = DEFAULT_TEST_TIMEOUT) -> Dict[str, Any]:
    """Wait for a task to complete.
    
    Args:
        coordinator_url: URL of the coordinator server
        task_id: ID of the task to wait for
        timeout: Maximum time to wait in seconds
        
    Returns:
        Dict with task result if successful, None otherwise
    """
    import requests
    
    start_time = time.time()
    poll_interval = 2  # seconds
    
    while (time.time() - start_time) < timeout:
        try:
            # Check task status
            api_url = f"{coordinator_url.rstrip('/')}/api/tasks/{task_id}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                result = response.json()
                if not result.get("success"):
                    logger.error(f"Error checking task status: {result.get('error')}")
                    return None
                    
                task_data = result.get("task")
                if not task_data:
                    logger.error(f"No task data returned for {task_id}")
                    return None
                    
                status = task_data.get("status")
                
                if status == "completed":
                    logger.info(f"Task {task_id} completed successfully")
                    return task_data
                elif status == "failed":
                    logger.error(f"Task {task_id} failed: {task_data.get('error')}")
                    return task_data
                elif status in ["timed_out", "canceled"]:
                    logger.warning(f"Task {task_id} {status}")
                    return task_data
                else:
                    # Still running or queued
                    logger.info(f"Task {task_id} is {status}... waiting")
            else:
                logger.error(f"Error checking task status: HTTP {response.status_code}")
                
            # Wait before polling again
            time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Error checking task status: {e}")
            time.sleep(poll_interval)
    
    # Timeout
    logger.error(f"Timeout waiting for task {task_id} completion")
    return None


def generate_security_config(file_path: str = DEFAULT_SECURITY_CONFIG) -> Dict[str, Any]:
    """Generate security configuration with API keys.
    
    Args:
        file_path: Path to save the security configuration file
        
    Returns:
        Dict with security configuration
    """
    # Generate a random token secret
    token_secret = str(uuid.uuid4())
    
    # Generate a worker API key
    worker_api_key = f"wk_{uuid.uuid4().hex}"
    
    # Create configuration
    config = {
        "token_secret": token_secret,
        "api_keys": {
            "worker": worker_api_key
        }
    }
    
    # Save to file
    try:
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Generated security config at {file_path}")
    except Exception as e:
        logger.error(f"Error writing security config: {e}")
    
    return config


def run_all_mode(host: str, port: int, dashboard_port: int, db_path: Optional[str] = None,
               worker_count: int = DEFAULT_WORKER_COUNT) -> List[subprocess.Popen]:
    """Run all components (coordinator, workers, dashboard) for testing.
    
    Args:
        host: Host to bind servers to
        port: Port for coordinator
        dashboard_port: Port for dashboard
        db_path: Optional path to DuckDB database
        worker_count: Number of worker nodes to start
        
    Returns:
        List of subprocess objects
    """
    processes = []
    
    # Generate security config
    security_file = os.path.join(tempfile.gettempdir(), "distributed_testing_security.json")
    security_config = generate_security_config(security_file)
    
    # Start coordinator
    coordinator_url = f"http://{host}:{port}"
    coordinator_process = run_coordinator(host, port, db_path, security_file)
    if coordinator_process:
        processes.append(coordinator_process)
        
    # Wait for coordinator to start
    time.sleep(2)
    
    # Start workers
    worker_api_key = security_config["api_keys"]["worker"]
    for i in range(worker_count):
        worker_id = f"worker_{i+1}"
        worker_dir = os.path.join(tempfile.gettempdir(), f"worker_{i+1}")
        os.makedirs(worker_dir, exist_ok=True)
        
        worker_process = run_worker(coordinator_url, worker_api_key, worker_id, worker_dir)
        if worker_process:
            processes.append(worker_process)
            
        # Slight delay between worker starts
        time.sleep(0.5)
    
    # Start dashboard
    dashboard_process = run_dashboard(host, dashboard_port, coordinator_url, auto_open=True)
    if dashboard_process:
        processes.append(dashboard_process)
    
    # Return all processes
    return processes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed Testing Framework Test Runner")
    
    parser.add_argument("--mode", choices=[
                      MODE_COORDINATOR, MODE_WORKER, MODE_CLIENT, MODE_DASHBOARD, MODE_ALL
                      ], default=MODE_ALL,
                      help="Mode to run in")
    
    # Coordinator options
    parser.add_argument("--host", default=DEFAULT_HOST,
                      help="Host to bind servers to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                      help="Port for the coordinator (or API in client mode)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH,
                      help="Path to DuckDB database (in-memory if not specified)")
    parser.add_argument("--security-config", default=DEFAULT_SECURITY_CONFIG,
                      help="Path to security configuration file")
    
    # Worker options
    parser.add_argument("--coordinator", default=None,
                      help="URL of the coordinator server (for worker and client modes)")
    parser.add_argument("--api-key", default=None,
                      help="API key for authentication (for worker mode)")
    parser.add_argument("--worker-id", default=None,
                      help="Worker ID (for worker mode, generated if not provided)")
    parser.add_argument("--work-dir", default=None,
                      help="Working directory for tasks (for worker mode)")
    
    # Dashboard options
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT,
                      help="Port for the dashboard server")
    parser.add_argument("--dashboard-auto-open", action="store_true",
                      help="Automatically open dashboard in web browser")
    
    # Client options
    parser.add_argument("--test-file", default=None,
                      help="Test file to submit (for client mode)")
    parser.add_argument("--test-args", default=None,
                      help="Arguments for the test (for client mode)")
    parser.add_argument("--priority", type=int, default=5,
                      help="Priority of the task (for client mode, lower is higher)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TEST_TIMEOUT,
                      help="Timeout in seconds (for client mode)")
    
    # All mode options
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT,
                      help="Number of worker nodes to start (for all mode)")
    
    args = parser.parse_args()
    
    try:
        # Handle different modes
        if args.mode == MODE_COORDINATOR:
            # Run coordinator
            run_coordinator(args.host, args.port, args.db_path, args.security_config)
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Coordinator interrupted by user")
                
        elif args.mode == MODE_WORKER:
            # Check required arguments
            if not args.coordinator:
                logger.error("Coordinator URL is required in worker mode")
                return 1
                
            if not args.api_key:
                logger.error("API key is required in worker mode")
                return 1
                
            # Run worker
            run_worker(args.coordinator, args.api_key, args.worker_id, args.work_dir)
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Worker interrupted by user")
                
        elif args.mode == MODE_DASHBOARD:
            # Check required arguments
            if not args.coordinator:
                logger.error("Coordinator URL is required in dashboard mode")
                return 1
                
            # Run dashboard
            run_dashboard(args.host, args.dashboard_port, args.coordinator, args.dashboard_auto_open)
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Dashboard interrupted by user")
                
        elif args.mode == MODE_CLIENT:
            # Check required arguments
            if not args.coordinator:
                logger.error("Coordinator URL is required in client mode")
                return 1
                
            if not args.test_file:
                logger.error("Test file is required in client mode")
                return 1
                
            # Parse test args
            test_args = args.test_args.split() if args.test_args else []
            
            # Submit task
            task_id = submit_test_task(args.coordinator, args.test_file, test_args, args.priority)
            if not task_id:
                logger.error("Failed to submit test task")
                return 1
                
            # Wait for completion
            result = wait_for_task_completion(args.coordinator, task_id, args.timeout)
            if not result:
                logger.error("Failed to get task result")
                return 1
                
            # Check result
            if result.get("status") == "completed":
                logger.info("Test completed successfully")
                return 0
            else:
                logger.error(f"Test failed with status: {result.get('status')}")
                return 1
                
        elif args.mode == MODE_ALL:
            # Run all components
            processes = run_all_mode(
                args.host, args.port, args.dashboard_port,
                args.db_path, args.worker_count
            )
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("All components interrupted by user")
                
                # Stop all processes
                for process in processes:
                    if process:
                        process.terminate()
                        
                for process in processes:
                    if process:
                        process.wait()
        
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())