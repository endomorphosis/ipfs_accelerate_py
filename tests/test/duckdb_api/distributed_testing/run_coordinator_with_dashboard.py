#!/usr/bin/env python3
"""
Run Coordinator with Load Balancer and Monitoring Dashboard

This script provides a complete solution for running the Distributed Testing Framework
with the Coordinator, Load Balancer, and Monitoring Dashboard components integrated.

Features:
- Configurable Coordinator settings
- Customizable Load Balancer with multiple scheduler types
- Real-time Monitoring Dashboard with web interface
- Terminal-based Monitoring Dashboard option
- Support for security configuration (API keys)
- Automatic worker registration
- Performance visualization
- Stress testing integration
- Health checking and diagnostics

Usage examples:
    # Basic usage with default settings
    python run_coordinator_with_dashboard.py

    # Custom ports and database
    python run_coordinator_with_dashboard.py --port 8888 --dashboard-port 5555 --db-path ./my_database.duckdb

    # Advanced load balancer configuration
    python run_coordinator_with_dashboard.py --scheduler composite --monitoring-interval 5 --rebalance-interval 60

    # Run with terminal-based dashboard (no web interface)
    python run_coordinator_with_dashboard.py --terminal-dashboard

    # Run with mock workers for testing
    python run_coordinator_with_dashboard.py --mock-workers 5

    # Run with stress testing enabled
    python run_coordinator_with_dashboard.py --stress-test --test-workers 10 --test-tasks 50
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
import threading
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_with_dashboard")

def generate_default_security_config(output_path=None):
    """Generate a default security configuration."""
    import secrets
    
    config = {
        "token_secret": secrets.token_hex(32),
        "api_keys": {
            "admin": secrets.token_hex(16),
            "worker": secrets.token_hex(16),
            "user": secrets.token_hex(16)
        }
    }
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
    return config

def start_terminal_dashboard(coordinator, load_balancer, refresh_interval=1.0):
    """Start a terminal-based dashboard for monitoring."""
    try:
        from duckdb_api.distributed_testing.load_balancer_live_dashboard import TerminalDashboard
        
        # Create and start the terminal dashboard
        dashboard = TerminalDashboard(
            coordinator=coordinator,
            load_balancer=load_balancer,
            refresh_interval=refresh_interval
        )
        
        # Start dashboard in new thread
        dashboard_thread = threading.Thread(target=dashboard.start)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        return dashboard
    except ImportError as e:
        logger.error(f"Failed to start terminal dashboard: {e}")
        logger.error("Make sure you have the required packages installed:")
        logger.error("pip install blessed numpy")
        return None

def launch_mock_workers(coordinator_url, api_key, count=3, capabilities=None):
    """Launch mock worker processes for testing."""
    worker_processes = []
    
    # Create temporary directory for worker files
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="mock_workers_")
    
    # Default capabilities for different worker types
    default_capabilities = [
        # Basic CPU worker
        {
            "hardware_types": ["cpu"],
            "memory_gb": 4
        },
        # CPU + GPU worker
        {
            "hardware_types": ["cpu", "cuda"],
            "cuda_compute": 7.5,
            "memory_gb": 16
        },
        # CPU + WebGPU worker
        {
            "hardware_types": ["cpu", "webgpu"],
            "browsers": ["chrome", "firefox"],
            "memory_gb": 8
        },
        # CPU + TPU worker
        {
            "hardware_types": ["cpu", "tpu"],
            "tpu_version": "v3",
            "memory_gb": 32
        },
        # Low-powered CPU worker
        {
            "hardware_types": ["cpu"],
            "cpu_cores": 2,
            "memory_gb": 2
        }
    ]
    
    # Override with custom capabilities if provided
    if capabilities:
        default_capabilities = capabilities
    
    # Launch worker processes
    for i in range(count):
        worker_id = f"mock_worker_{i}"
        worker_dir = os.path.join(temp_dir, f"worker_{i}")
        os.makedirs(worker_dir, exist_ok=True)
        
        # Select capability set (cycling through the available options)
        capability_index = i % len(default_capabilities)
        capabilities_json = json.dumps(default_capabilities[capability_index])
        
        worker_cmd = [
            sys.executable,
            os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
            "--coordinator", coordinator_url,
            "--api-key", api_key,
            "--worker-id", worker_id,
            "--work-dir", worker_dir,
            "--reconnect-interval", "2",
            "--heartbeat-interval", "3",
            "--capabilities", capabilities_json
        ]
        
        logger.info(f"Starting mock worker {worker_id} with capabilities: {default_capabilities[capability_index]}")
        process = subprocess.Popen(worker_cmd)
        worker_processes.append({
            "process": process,
            "worker_id": worker_id,
            "work_dir": worker_dir,
            "capabilities": default_capabilities[capability_index]
        })
    
    return worker_processes, temp_dir

def run_stress_test(coordinator, task_count=20, duration=60, task_types=None):
    """Run a stress test with configurable parameters."""
    logger.info(f"Starting stress test with {task_count} tasks for {duration} seconds")
    
    # Default task types if none provided
    if not task_types:
        task_types = [
            {
                "type": "command",
                "config": {"command": ["sleep", "5"]},
                "requirements": {"hardware": ["cpu"]},
                "priority": 1
            },
            {
                "type": "command",
                "config": {"command": ["echo", "GPU task"]},
                "requirements": {"hardware": ["cuda"]},
                "priority": 2
            },
            {
                "type": "command",
                "config": {"command": ["echo", "WebGPU task"]},
                "requirements": {"hardware": ["webgpu"]},
                "priority": 3
            },
            {
                "type": "benchmark",
                "config": {
                    "model": "bert-base-uncased",
                    "batch_sizes": [1, 2, 4],
                    "precision": "fp16",
                    "iterations": 3
                },
                "requirements": {"hardware": ["cpu"], "min_memory_gb": 8},
                "priority": 1
            },
            {
                "type": "benchmark",
                "config": {
                    "model": "t5-small",
                    "batch_sizes": [1],
                    "precision": "fp32",
                    "iterations": 2
                },
                "requirements": {"hardware": ["cpu"]},
                "priority": 4
            }
        ]
    
    # Create tasks
    task_ids = []
    start_time = time.time()
    
    # Submit initial batch of tasks
    for i in range(task_count // 2):  # Submit half initially
        task_type = task_types[i % len(task_types)]
        task_id = coordinator.add_task(
            task_type["type"],
            task_type["config"],
            task_type["requirements"],
            task_type["priority"]
        )
        task_ids.append(task_id)
    
    # Continue submitting tasks at a steady rate until duration is reached
    remaining_tasks = task_count - (task_count // 2)
    task_interval = duration / (remaining_tasks + 1)  # Spread over the duration
    
    def submit_remaining_tasks():
        nonlocal remaining_tasks
        next_task_index = task_count // 2
        
        while time.time() - start_time < duration and remaining_tasks > 0:
            # Submit next task
            task_type = task_types[next_task_index % len(task_types)]
            task_id = coordinator.add_task(
                task_type["type"],
                task_type["config"],
                task_type["requirements"],
                task_type["priority"]
            )
            task_ids.append(task_id)
            
            # Update counters
            next_task_index += 1
            remaining_tasks -= 1
            
            # Wait before submitting next task
            time.sleep(task_interval)
    
    # Start task submission in separate thread
    submission_thread = threading.Thread(target=submit_remaining_tasks)
    submission_thread.daemon = True
    submission_thread.start()
    
    return task_ids

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Coordinator with Load Balancer and Monitoring Dashboard")
    
    # Coordinator settings
    coordinator_group = parser.add_argument_group("Coordinator Settings")
    coordinator_group.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    coordinator_group.add_argument("--port", type=int, default=8080, help="Port to bind to")
    coordinator_group.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to DuckDB database")
    coordinator_group.add_argument("--heartbeat-timeout", type=int, default=60, help="Heartbeat timeout in seconds")
    coordinator_group.add_argument("--security-config", type=str, help="Path to security configuration JSON file")
    coordinator_group.add_argument("--visualization-path", type=str, help="Path for performance visualizations")
    
    # Load Balancer settings
    lb_group = parser.add_argument_group("Load Balancer Settings")
    lb_group.add_argument("--disable-load-balancer", action="store_true", help="Disable load balancer integration")
    lb_group.add_argument("--scheduler", type=str, default="performance_based", 
                         choices=["performance_based", "round_robin", "weighted_round_robin", "priority_based", "affinity_based", "composite"],
                         help="Load balancer scheduler type")
    lb_group.add_argument("--monitoring-interval", type=int, default=15, help="Monitoring interval in seconds")
    lb_group.add_argument("--rebalance-interval", type=int, default=90, help="Rebalance interval in seconds")
    lb_group.add_argument("--enable-work-stealing", action="store_true", help="Enable work stealing between workers")
    lb_group.add_argument("--worker-concurrency", type=int, default=2, help="Default worker concurrency setting")
    
    # Dashboard settings
    dashboard_group = parser.add_argument_group("Dashboard Settings")
    dashboard_group.add_argument("--disable-dashboard", action="store_true", help="Disable all monitoring dashboards")
    dashboard_group.add_argument("--dashboard-port", type=int, default=5000, help="Port for web monitoring dashboard")
    dashboard_group.add_argument("--metrics-db-path", type=str, help="Path to metrics database (defaults to <db-path>_metrics.duckdb)")
    dashboard_group.add_argument("--terminal-dashboard", action="store_true", help="Use terminal-based dashboard instead of web interface")
    dashboard_group.add_argument("--terminal-refresh", type=float, default=1.0, help="Terminal dashboard refresh interval in seconds")
    dashboard_group.add_argument("--open-browser", action="store_true", help="Open web browser to dashboard automatically")
    
    # Testing options
    testing_group = parser.add_argument_group("Testing Options")
    testing_group.add_argument("--mock-workers", type=int, help="Launch N mock workers for testing")
    testing_group.add_argument("--stress-test", action="store_true", help="Run a stress test after starting")
    testing_group.add_argument("--test-tasks", type=int, default=20, help="Number of tasks to create for stress test")
    testing_group.add_argument("--test-duration", type=int, default=60, help="Duration of stress test in seconds")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default metrics DB path if not specified
    if not args.metrics_db_path and args.db_path:
        metrics_db_base = os.path.splitext(args.db_path)[0]
        args.metrics_db_path = f"{metrics_db_base}_metrics.duckdb"
    
    # Load or generate security configuration
    security_config = None
    security_config_path = args.security_config
    
    if security_config_path and os.path.exists(security_config_path):
        # Load existing configuration
        with open(security_config_path, 'r') as f:
            security_config = json.load(f)
        logger.info(f"Loaded security configuration from {security_config_path}")
    else:
        # Generate new configuration
        if not security_config_path:
            security_config_path = os.path.join(os.path.dirname(args.db_path), "security_config.json")
        
        security_config = generate_default_security_config(security_config_path)
        logger.info(f"Generated new security configuration at {security_config_path}")
    
    # Apply the patches to integrate the load balancer
    try:
        # Import coordinator patch (applies patches automatically)
        from duckdb_api.distributed_testing.coordinator_patch import apply_patches, remove_patches
        
        # Apply patches if load balancer is enabled
        if not args.disable_load_balancer:
            apply_patches()
            logger.info("Applied coordinator load balancer integration patches")
    except ImportError:
        logger.error("Failed to import coordinator_patch module. Make sure it exists in the distributed_testing directory.")
        sys.exit(1)
    
    # Import coordinator 
    try:
        from duckdb_api.distributed_testing.coordinator import CoordinatorServer
    except ImportError:
        logger.error("Failed to import CoordinatorServer. Make sure it exists in the distributed_testing directory.")
        sys.exit(1)
    
    # Load balancer configuration
    load_balancer_config = {
        "db_path": args.db_path,
        "monitoring_interval": args.monitoring_interval,
        "rebalance_interval": args.rebalance_interval,
        "default_scheduler": {
            "type": args.scheduler
        },
        "worker_concurrency": args.worker_concurrency,
        "enable_work_stealing": args.enable_work_stealing,
        "test_type_schedulers": {
            "performance": {"type": "performance_based"},
            "compatibility": {"type": "affinity_based"},
            "integration": {
                "type": "composite",
                "algorithms": [
                    {"type": "performance_based", "weight": 0.7},
                    {"type": "priority_based", "weight": 0.3}
                ]
            }
        }
    }
    
    # Configure test type to scheduler mapping based on model family
    load_balancer_config["model_family_schedulers"] = {
        "vision": {"type": "performance_based"},
        "text": {"type": "weighted_round_robin"},
        "audio": {"type": "affinity_based"},
        "multimodal": {
            "type": "composite",
            "algorithms": [
                {"type": "performance_based", "weight": 0.6},
                {"type": "affinity_based", "weight": 0.4}
            ]
        }
    }
    
    # Create and start coordinator
    try:
        # Ensure the database directory exists
        db_dir = os.path.dirname(os.path.abspath(args.db_path))
        os.makedirs(db_dir, exist_ok=True)
        
        # Create coordinator
        coordinator = CoordinatorServer(
            host=args.host,
            port=args.port,
            db_path=args.db_path,
            heartbeat_timeout=args.heartbeat_timeout,
            visualization_path=args.visualization_path,
            performance_analyzer=True,
            enable_load_balancer=not args.disable_load_balancer,
            load_balancer_config=load_balancer_config,
            token_secret=security_config["token_secret"]
        )
        
        # Start coordinator in a separate thread
        coordinator_thread = threading.Thread(target=lambda: asyncio.run(coordinator.start()))
        coordinator_thread.daemon = True
        coordinator_thread.start()
        
        # Wait for coordinator to initialize
        time.sleep(2)
        
        logger.info(f"Coordinator started on ws://{args.host}:{args.port} with load balancer {'enabled' if not args.disable_load_balancer else 'disabled'}")
        logger.info(f"Admin API Key: {security_config['api_keys']['admin']}")
        logger.info(f"Worker API Key: {security_config['api_keys']['worker']}")
        logger.info(f"User API Key: {security_config['api_keys']['user']}")
        
        # Start monitoring dashboard if enabled
        dashboard_integration = None
        terminal_dashboard = None
        
        if not args.disable_dashboard:
            if args.terminal_dashboard:
                # Start terminal-based dashboard
                terminal_dashboard = start_terminal_dashboard(
                    coordinator=coordinator, 
                    load_balancer=coordinator.load_balancer if hasattr(coordinator, 'load_balancer') else None,
                    refresh_interval=args.terminal_refresh
                )
                
                if terminal_dashboard:
                    logger.info(f"Terminal dashboard started with refresh interval of {args.terminal_refresh}s")
            else:
                # Start web-based dashboard
                try:
                    # Import monitoring integration
                    from duckdb_api.distributed_testing.load_balancer.monitoring.integration import MonitoringIntegration
                    
                    # Create monitoring integration
                    dashboard_integration = MonitoringIntegration(
                        coordinator=coordinator,
                        load_balancer=coordinator.load_balancer if hasattr(coordinator, 'load_balancer') else None,
                        db_path=args.metrics_db_path,
                        dashboard_host=args.host,
                        dashboard_port=args.dashboard_port,
                        collection_interval=1.0  # Collect metrics every second
                    )
                    
                    # Start monitoring
                    dashboard_integration.start()
                    
                    logger.info(f"Web monitoring dashboard started at http://{args.host}:{args.dashboard_port}/")
                    
                    # Open browser if requested
                    if args.open_browser:
                        url = f"http://{args.host}:{args.dashboard_port}/"
                        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
                
                except ImportError as e:
                    logger.error(f"Failed to start monitoring dashboard: {e}")
                    logger.error("Make sure you have Flask and Flask-SocketIO installed:")
                    logger.error("pip install flask flask-cors flask-socketio")
        
        # Start mock workers if requested
        mock_workers = []
        temp_dir = None
        
        if args.mock_workers and args.mock_workers > 0:
            coordinator_url = f"ws://{args.host}:{args.port}"
            mock_workers, temp_dir = launch_mock_workers(
                coordinator_url=coordinator_url,
                api_key=security_config["api_keys"]["worker"],
                count=args.mock_workers
            )
            
            logger.info(f"Started {len(mock_workers)} mock workers")
            
            # Wait for workers to register
            time.sleep(5)
        
        # Run stress test if requested
        stress_test_tasks = []
        if args.stress_test:
            stress_test_tasks = run_stress_test(
                coordinator=coordinator,
                task_count=args.test_tasks,
                duration=args.test_duration
            )
            
            logger.info(f"Started stress test with {len(stress_test_tasks)} tasks")
        
        # Keep running until interrupted
        try:
            logger.info("Press Ctrl+C to stop the server")
            
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            
            # Stop dashboard if running
            if dashboard_integration:
                logger.info("Stopping web dashboard...")
                dashboard_integration.stop()
            
            if terminal_dashboard:
                logger.info("Stopping terminal dashboard...")
                terminal_dashboard.stop()
            
            # Terminate mock workers
            if mock_workers:
                logger.info(f"Terminating {len(mock_workers)} mock workers...")
                for worker in mock_workers:
                    worker["process"].terminate()
                
                # Wait for workers to terminate
                for worker in mock_workers:
                    worker["process"].wait()
                
                # Clean up temporary directory
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
            
            # Stop coordinator
            logger.info("Stopping coordinator...")
            asyncio.run(coordinator.stop())
            
            logger.info("Shutdown complete")
    
    except Exception as e:
        logger.error(f"Error running coordinator: {e}")
        import traceback
        traceback.print_exc()
        
        # Remove patches if error occurs
        if not args.disable_load_balancer:
            remove_patches()

if __name__ == "__main__":
    main()