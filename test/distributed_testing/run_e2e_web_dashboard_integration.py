#!/usr/bin/env python3
"""
End-to-End Integration Test for Result Aggregator Web Dashboard

This script performs a comprehensive end-to-end test of the Result Aggregator Web Dashboard
with the full Distributed Testing Framework. It:
1. Sets up a complete distributed testing environment (coordinator and workers)
2. Generates real-world test workloads
3. Starts the web dashboard to visualize the results
4. Tests real-time notifications
5. Verifies all dashboard components

Usage:
    python run_e2e_web_dashboard_integration.py [--db-path DB_PATH] [--coordinator-port PORT] 
                                               [--dashboard-port PORT] [--num-workers NUM]
                                               [--num-tasks NUM] [--debug]
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Callable

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import framework components
from coordinator import DistributedTestingCoordinator
from worker import DistributedTestingWorker
from result_aggregator.service import ResultAggregatorService
from result_aggregator.coordinator_integration import ResultAggregatorIntegration
from result_aggregator.web_dashboard import app, main as run_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("e2e_integration_test.log")
    ]
)
logger = logging.getLogger("e2e_integration_test")

# Test task types
TASK_TYPES = [
    "benchmark", 
    "unit_test", 
    "integration_test", 
    "performance_test",
    "compatibility_test"
]

# Test model names
MODEL_NAMES = [
    "bert-base-uncased",
    "t5-small",
    "vit-base-patch16-224",
    "whisper-tiny",
    "llama-7b",
    "clip-vit-base-patch32",
    "wav2vec2-base",
    "gpt2"
]

# Hardware types
HARDWARE_TYPES = [
    "cpu",
    "cuda",
    "rocm",
    "mps",
    "webgpu",
    "webnn"
]

# Global test state
running = True
coordinator_process = None
workers = []
dashboard_process = None
integration = None

def notification_callback(notification):
    """Handle notifications from the Result Aggregator"""
    logger.info(f"NOTIFICATION: {notification['type'].upper()} - {notification['severity'].upper()}")
    logger.info(f"Message: {notification['message']}")
    logger.info(f"Timestamp: {notification['timestamp']}")
    
    if "details" in notification and notification["details"]:
        if notification["type"] == "anomaly":
            if "anomalous_features" in notification["details"]:
                for feature in notification["details"]["anomalous_features"]:
                    logger.info(f"  • {feature['feature']}: value={feature['value']:.2f}, z-score={feature['z_score']:.2f}")
        elif notification["type"] == "trend":
            logger.info(f"  • Metric: {notification['details']['metric']}")
            logger.info(f"  • Trend: {notification['details']['trend']}")
            logger.info(f"  • Percent Change: {notification['details']['percent_change']:.2f}%")

def is_port_available(port: int) -> bool:
    """Check if a port is available for use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_coordinator(args):
    """Start the coordinator process"""
    global coordinator_process
    
    # Check if port is available
    if not is_port_available(args.coordinator_port):
        logger.error(f"Port {args.coordinator_port} is already in use. Please choose a different port.")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(current_dir, "coordinator.py"),
        "--host", "0.0.0.0",
        "--port", str(args.coordinator_port),
        "--db-path", args.db_path
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    # Start coordinator
    logger.info(f"Starting coordinator: {' '.join(cmd)}")
    coordinator_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for coordinator to start
    time.sleep(2)
    
    # Check if coordinator started successfully
    if coordinator_process.poll() is not None:
        stderr = coordinator_process.stderr.read()
        logger.error(f"Failed to start coordinator: {stderr}")
        sys.exit(1)
    
    logger.info(f"Coordinator started on port {args.coordinator_port}")
    return coordinator_process

def start_worker(args, worker_id, capabilities):
    """Start a worker process"""
    global workers
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(current_dir, "worker.py"),
        "--coordinator", f"http://localhost:{args.coordinator_port}",
        "--worker-id", worker_id,
        "--capabilities", json.dumps(capabilities)
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    # Start worker
    logger.info(f"Starting worker {worker_id}: {' '.join(cmd)}")
    worker_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for worker to start
    time.sleep(1)
    
    # Check if worker started successfully
    if worker_process.poll() is not None:
        stderr = worker_process.stderr.read()
        logger.error(f"Failed to start worker {worker_id}: {stderr}")
        return None
    
    logger.info(f"Worker {worker_id} started")
    workers.append(worker_process)
    return worker_process

def start_dashboard(args):
    """Start the web dashboard process"""
    global dashboard_process
    
    # Check if port is available
    if not is_port_available(args.dashboard_port):
        logger.error(f"Port {args.dashboard_port} is already in use. Please choose a different port.")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(current_dir, "run_web_dashboard.py"),
        "--port", str(args.dashboard_port),
        "--db-path", args.db_path,
        "--update-interval", str(args.update_interval)
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    # Start dashboard
    logger.info(f"Starting web dashboard: {' '.join(cmd)}")
    dashboard_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for dashboard to start
    time.sleep(2)
    
    # Check if dashboard started successfully
    if dashboard_process.poll() is not None:
        stderr = dashboard_process.stderr.read()
        logger.error(f"Failed to start web dashboard: {stderr}")
        sys.exit(1)
    
    logger.info(f"Web dashboard started on port {args.dashboard_port} with WebSocket update interval {args.update_interval}s")
    return dashboard_process

def cleanup():
    """Clean up all processes"""
    logger.info("Cleaning up processes...")
    
    # Terminate all processes
    if dashboard_process:
        logger.info("Terminating dashboard")
        dashboard_process.terminate()
    
    for worker in workers:
        logger.info("Terminating worker")
        worker.terminate()
    
    if coordinator_process:
        logger.info("Terminating coordinator")
        coordinator_process.terminate()
    
    # Wait for all processes to terminate
    time.sleep(2)
    
    # Forcefully kill any remaining processes
    for process in [dashboard_process, *workers, coordinator_process]:
        if process and process.poll() is None:
            logger.info(f"Forcefully killing process {process.pid}")
            try:
                os.kill(process.pid, signal.SIGKILL)
            except:
                pass
    
    logger.info("Cleanup complete")

def generate_random_task(coordinator, task_id, task_type, hardware_requirements=None, priority=None):
    """Generate a random task for testing"""
    # Select random hardware requirements if not specified
    if hardware_requirements is None:
        hardware_count = random.randint(1, 3)
        hardware_requirements = random.sample(HARDWARE_TYPES, hardware_count)
    
    # Select random priority if not specified
    if priority is None:
        priority = random.randint(1, 5)
    
    # Select random model
    model = random.choice(MODEL_NAMES)
    
    # Generate task
    task = {
        "task_id": task_id,
        "type": task_type,
        "status": "pending",
        "priority": priority,
        "requirements": {"hardware": hardware_requirements},
        "metadata": {
            "model": model,
            "batch_size": random.choice([1, 2, 4, 8, 16, 32]),
            "precision": random.choice(["fp32", "fp16", "int8", "int4"])
        },
        "attempts": 0,
        "created": datetime.now().isoformat()
    }
    
    # Add to coordinator's tasks
    coordinator.tasks[task_id] = task
    return task

def generate_task_result(task, worker, success=True, execution_time=None, generate_anomaly=False):
    """Generate a result for a given task"""
    if execution_time is None:
        # Random execution time between 1 and 60 seconds
        base_time = random.uniform(1.0, 15.0)
        if task["type"] == "benchmark":
            # Benchmarks take longer
            base_time *= 4
        execution_time = base_time
    
    # Determine model based on task metadata
    model = task.get("metadata", {}).get("model", "unknown_model")
    batch_size = task.get("metadata", {}).get("batch_size", 1)
    precision = task.get("metadata", {}).get("precision", "fp32")
    
    if success:
        # Generate success result with metrics
        result = {
            "status": "success",
            "metrics": {}
        }
        
        # Generate normal or anomalous metrics
        if task["type"] == "benchmark":
            # Benchmark metrics
            if generate_anomaly and random.random() < 0.8:  # 80% chance of anomaly
                # Generate anomalous throughput (very high or very low)
                if random.random() < 0.5:
                    # Very high throughput (5-10x normal)
                    throughput = random.uniform(500, 1000)
                else:
                    # Very low throughput (0.1-0.3x normal)
                    throughput = random.uniform(10, 30)
                
                # Generate anomalous latency (inversely related to throughput)
                latency = 1000 / throughput
                
                # Anomalous memory usage (very high)
                memory_usage = random.uniform(8000, 16000)
                
                # Add anomalous metric values
                result["metrics"]["throughput"] = throughput
                result["metrics"]["latency_ms"] = latency
                result["metrics"]["memory_usage_mb"] = memory_usage
            else:
                # Normal metrics for benchmark
                # Base throughput depends on hardware
                base_throughput = 0
                if "cuda" in worker["capabilities"]["hardware"]:
                    base_throughput = random.uniform(80, 120)
                elif "rocm" in worker["capabilities"]["hardware"]:
                    base_throughput = random.uniform(70, 110)
                elif "webgpu" in worker["capabilities"]["hardware"]:
                    base_throughput = random.uniform(40, 80)
                else:  # CPU
                    base_throughput = random.uniform(20, 40)
                
                # Adjust for batch size
                throughput = base_throughput * (batch_size ** 0.7) / (execution_time ** 0.3)
                
                # Calculate latency (inversely related to throughput)
                latency = 1000 / throughput
                
                # Memory usage based on model and batch size
                memory_usage = random.uniform(500, 2000) * (batch_size ** 0.6)
                
                # Add normal metric values
                result["metrics"]["throughput"] = throughput
                result["metrics"]["latency_ms"] = latency
                result["metrics"]["memory_usage_mb"] = memory_usage
        
        elif task["type"] == "performance_test":
            # Performance test metrics
            if generate_anomaly and random.random() < 0.8:  # 80% chance of anomaly
                # Anomalous QPS (very high or very low)
                if random.random() < 0.5:
                    qps = random.uniform(5000, 10000)
                else:
                    qps = random.uniform(10, 50)
                
                # Anomalous response time
                response_time = random.uniform(500, 2000)
                
                # Anomalous CPU usage
                cpu_usage = random.uniform(90, 100)
            else:
                # Normal metrics for performance test
                qps = random.uniform(500, 2000)
                response_time = random.uniform(50, 200)
                cpu_usage = random.uniform(20, 80)
            
            # Add metric values
            result["metrics"]["qps"] = qps
            result["metrics"]["response_time_ms"] = response_time
            result["metrics"]["cpu_usage_percent"] = cpu_usage
        
        else:
            # Other test types - basic metrics
            if generate_anomaly and random.random() < 0.8:  # 80% chance of anomaly
                # Anomalous duration (very long)
                result["metrics"]["test_duration"] = execution_time * random.uniform(5, 10)
                result["metrics"]["assertion_count"] = random.randint(1, 10)
            else:
                # Normal metrics
                result["metrics"]["test_duration"] = execution_time
                result["metrics"]["assertion_count"] = random.randint(10, 100)
        
        # Add common metrics for all tasks
        result["metrics"]["execution_time"] = execution_time
        
        # Add details based on task type
        if task["type"] == "benchmark":
            result["details"] = {
                "model": model,
                "batch_size": batch_size,
                "precision": precision,
                "hardware": worker["capabilities"]["hardware"]
            }
        elif task["type"] in ["unit_test", "integration_test", "functional_test"]:
            result["details"] = {
                "passed_tests": random.randint(10, 100),
                "failed_tests": 0,
                "skipped_tests": random.randint(0, 5)
            }
        elif task["type"] == "performance_test":
            result["details"] = {
                "test_iterations": random.randint(100, 1000),
                "samples": random.randint(1000, 10000),
                "warmup_iterations": random.randint(10, 50)
            }
    else:
        # Generate failure result with error message
        error_types = [
            "RuntimeError: Out of memory",
            "AssertionError: Test failed",
            "TimeoutError: Operation timed out",
            "ValueError: Invalid input parameters",
            "ResourceError: Required resource not available"
        ]
        
        error_message = random.choice(error_types)
        
        result = {
            "status": "failed",
            "error": error_message,
            "details": {
                "stack_trace": f"Traceback (most recent call last):\n  File \"example.py\", line {random.randint(10, 500)}, in run_test\n    {error_message}"
            }
        }
    
    return result, execution_time

async def simulate_tasks(coordinator, integration, args):
    """Simulate task creation, execution, and result reporting"""
    logger.info(f"Simulating {args.num_tasks} tasks...")
    
    # Create workers dictionary
    workers_dict = {}
    for i in range(args.num_workers):
        worker_id = f"worker-{i+1}"
        
        # Generate random hardware capabilities
        hw_count = random.randint(1, 4)
        hardware = random.sample(HARDWARE_TYPES, hw_count)
        
        workers_dict[worker_id] = {
            "worker_id": worker_id,
            "hostname": f"test-worker-{i+1}.example.com",
            "capabilities": {"hardware": hardware},
            "status": "active",
            "tasks_completed": 0,
            "tasks_failed": 0,
            "registered": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat()
        }
        
        logger.info(f"Created worker {worker_id} with hardware: {hardware}")
    
    # Set workers in coordinator
    coordinator.workers = workers_dict
    
    # Create and execute tasks
    tasks = []
    completed_count = 0
    failed_count = 0
    
    for i in range(args.num_tasks):
        # Generate task ID and type
        task_id = f"task-{i+1}"
        task_type = random.choice(TASK_TYPES)
        
        # Create task
        task = generate_random_task(coordinator, task_id, task_type)
        tasks.append(task)
        
        # Find compatible worker
        compatible_workers = []
        for worker_id, worker in workers_dict.items():
            # Check if worker has all required hardware
            required_hardware = task["requirements"]["hardware"]
            worker_hardware = worker["capabilities"]["hardware"]
            
            if any(hw in worker_hardware for hw in required_hardware):
                compatible_workers.append((worker_id, worker))
        
        if not compatible_workers:
            logger.warning(f"No compatible worker found for task {task_id}")
            continue
        
        # Select random compatible worker
        worker_id, worker = random.choice(compatible_workers)
        
        # Associate task with worker
        coordinator.running_tasks[task_id] = worker_id
        
        # Update task status
        task["status"] = "running"
        task["started"] = datetime.now().isoformat()
        task["attempts"] += 1
        
        # Simulate execution time (add some delay)
        await asyncio.sleep(0.05)
        
        # Determine success or failure
        success = random.random() < 0.9  # 90% success rate
        generate_anomaly = args.generate_anomalies and random.random() < 0.2  # 20% anomaly rate if enabled
        
        # Generate result
        result, execution_time = generate_task_result(
            task, 
            worker, 
            success=success,
            generate_anomaly=generate_anomaly
        )
        
        try:
            if success:
                # Call task completion handler
                await coordinator._handle_task_completed(task_id, worker_id, result, execution_time)
                completed_count += 1
                
                logger.info(f"Task {task_id} completed by worker {worker_id} in {execution_time:.2f}s")
            else:
                # Call task failure handler
                await coordinator._handle_task_failed(task_id, worker_id, result["error"], execution_time)
                failed_count += 1
                
                logger.info(f"Task {task_id} failed on worker {worker_id}: {result['error']}")
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
        
        # Add small delay between tasks
        await asyncio.sleep(0.05)
    
    logger.info(f"Task simulation complete: {completed_count} completed, {failed_count} failed")
    
    # Generate trend data by creating related tasks with trending metrics
    if args.generate_trends:
        logger.info("Generating performance trend data...")
        
        # Create trending tasks for a specific model and worker
        task_type = "benchmark"
        worker_id = list(workers_dict.keys())[0]
        worker = workers_dict[worker_id]
        
        # Generate 20 tasks with gradually increasing/decreasing metrics
        num_trend_tasks = 20
        
        # Base metrics
        base_throughput = 100.0
        base_latency = 10.0
        base_memory = 1000.0
        trend_direction = random.choice(["increasing", "decreasing"])
        
        for i in range(num_trend_tasks):
            task_id = f"trend-task-{i+1}"
            
            # Create task with consistent metadata
            task = generate_random_task(
                coordinator, 
                task_id, 
                task_type, 
                hardware_requirements=worker["capabilities"]["hardware"][:1]
            )
            
            # Use consistent metadata for trending
            task["metadata"] = {
                "model": "trending_model",
                "batch_size": 8,
                "precision": "fp16"
            }
            
            # Associate task with worker
            coordinator.running_tasks[task_id] = worker_id
            
            # Update task status
            task["status"] = "running"
            task["started"] = datetime.now().isoformat()
            task["attempts"] += 1
            
            # Calculate trend factor
            if trend_direction == "increasing":
                trend_factor = 1.0 + (i / num_trend_tasks) * 0.5  # 50% increase over the trend
            else:
                trend_factor = 1.0 - (i / num_trend_tasks) * 0.3  # 30% decrease over the trend
            
            # Create execution time and result with trending metrics
            execution_time = 10.0
            
            result = {
                "status": "success",
                "metrics": {
                    "throughput": base_throughput * trend_factor,
                    "latency_ms": base_latency / trend_factor,
                    "memory_usage_mb": base_memory * (trend_factor ** 0.3),
                    "execution_time": execution_time
                },
                "details": {
                    "model": task["metadata"]["model"],
                    "batch_size": task["metadata"]["batch_size"],
                    "precision": task["metadata"]["precision"],
                    "hardware": worker["capabilities"]["hardware"]
                }
            }
            
            try:
                # Call task completion handler
                await coordinator._handle_task_completed(task_id, worker_id, result, execution_time)
                logger.info(f"Trend task {task_id} completed with throughput {result['metrics']['throughput']:.2f}")
            except Exception as e:
                logger.error(f"Error processing trend task {task_id}: {e}")
            
            # Add a small delay between tasks
            await asyncio.sleep(0.05)
        
        logger.info(f"Generated {num_trend_tasks} trending tasks with {trend_direction} throughput")
    
    # Wait for all analysis to complete
    await asyncio.sleep(2)
    
    logger.info("Running additional analysis...")
    
    # Run analysis to generate reports
    anomalies = integration.service.detect_anomalies()
    logger.info(f"Detected {len(anomalies)} anomalies")
    
    # Generate performance trends
    trends = integration.service.analyze_performance_trends()
    significant_trends = []
    for metric, trend_data in trends.items():
        if "trend" in trend_data and trend_data["trend"] != "stable" and abs(trend_data.get("percent_change", 0)) > 5:
            significant_trends.append((metric, trend_data))
    
    logger.info(f"Detected {len(significant_trends)} significant trends")
    
    # Generate reports
    reports_dir = os.path.join(current_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate summary report
    summary_report = integration.service.generate_analysis_report(
        report_type="summary",
        format="markdown"
    )
    
    with open(os.path.join(reports_dir, "summary_report.md"), "w") as f:
        f.write(summary_report)
    
    # Generate performance report
    performance_report = integration.service.generate_analysis_report(
        report_type="performance",
        format="markdown"
    )
    
    with open(os.path.join(reports_dir, "performance_report.md"), "w") as f:
        f.write(performance_report)
    
    # Generate anomaly report if anomalies exist
    if anomalies:
        anomaly_report = integration.service.generate_analysis_report(
            report_type="anomaly",
            format="markdown"
        )
        
        with open(os.path.join(reports_dir, "anomaly_report.md"), "w") as f:
            f.write(anomaly_report)
    
    logger.info(f"Generated reports in {reports_dir}")
    
    # Open dashboard in web browser if requested
    if args.open_browser:
        dashboard_url = f"http://localhost:{args.dashboard_port}"
        logger.info(f"Opening dashboard in web browser: {dashboard_url}")
        webbrowser.open(dashboard_url)
    
    return {
        "tasks": len(tasks),
        "completed": completed_count,
        "failed": failed_count,
        "anomalies": len(anomalies),
        "trends": len(significant_trends)
    }

async def run_test(args):
    """Run the end-to-end integration test"""
    global running, integration
    
    try:
        # Print test parameters
        logger.info("=" * 80)
        logger.info("END-TO-END WEB DASHBOARD INTEGRATION TEST")
        logger.info("=" * 80)
        logger.info(f"Database Path: {args.db_path}")
        logger.info(f"Coordinator Port: {args.coordinator_port}")
        logger.info(f"Dashboard Port: {args.dashboard_port}")
        logger.info(f"Number of Workers: {args.num_workers}")
        logger.info(f"Number of Tasks: {args.num_tasks}")
        logger.info(f"WebSocket Update Interval: {args.update_interval} seconds")
        logger.info(f"Generate Anomalies: {args.generate_anomalies}")
        logger.info(f"Generate Trends: {args.generate_trends}")
        logger.info(f"Debug Mode: {args.debug}")
        logger.info(f"Open Browser: {args.open_browser}")
        logger.info("=" * 80)
        
        # Initialize directly for in-process integration test
        logger.info("Initializing coordinator...")
        coordinator = DistributedTestingCoordinator(
            db_path=args.db_path,
            host="0.0.0.0",
            port=args.coordinator_port,
            enable_advanced_scheduler=True,
            enable_plugins=True
        )
        
        # Initialize result aggregator integration
        logger.info("Initializing result aggregator integration...")
        integration = ResultAggregatorIntegration(
            coordinator=coordinator,
            db_path=args.db_path,
            enable_ml=True,
            enable_visualization=True,
            enable_real_time_analysis=True,
            enable_notifications=True
        )
        
        # Register with coordinator
        integration.register_with_coordinator()
        
        # Register notification callback
        integration.register_notification_callback(notification_callback)
        
        # Start dashboard process
        start_dashboard(args)
        
        # Wait for dashboard to initialize
        logger.info("Waiting for dashboard to initialize...")
        await asyncio.sleep(2)
        
        # Simulate tasks and generate test data
        logger.info("Simulating tasks and generating test data...")
        results = await simulate_tasks(coordinator, integration, args)
        
        # Wait for dashboard to process all data
        logger.info("Waiting for dashboard to process all data...")
        await asyncio.sleep(2)
        
        # Display test summary
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Tasks Executed: {results['tasks']}")
        logger.info(f"Tasks Completed: {results['completed']}")
        logger.info(f"Tasks Failed: {results['failed']}")
        logger.info(f"Anomalies Detected: {results['anomalies']}")
        logger.info(f"Significant Trends: {results['trends']}")
        logger.info("=" * 80)
        
        # Display dashboard URL
        dashboard_url = f"http://localhost:{args.dashboard_port}"
        logger.info(f"Web Dashboard URL: {dashboard_url}")
        logger.info("Login Credentials:")
        logger.info("  Username: admin")
        logger.info("  Password: admin_password")
        
        # Keep running until interrupted
        logger.info("Test is now complete.")
        logger.info("Dashboard will remain running until you press Ctrl+C")
        
        while running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error running test: {e}", exc_info=True)
    finally:
        # Close the integration
        if integration:
            integration.close()
        
        # Clean up all processes
        cleanup()

def main():
    """Main function"""
    global running
    
    parser = argparse.ArgumentParser(description="End-to-End Integration Test for Result Aggregator Web Dashboard")
    parser.add_argument("--db-path", default="./e2e_test_results.duckdb", help="Path to DuckDB database")
    parser.add_argument("--coordinator-port", type=int, default=8081, help="Port for coordinator")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Port for web dashboard")
    parser.add_argument("--num-workers", type=int, default=5, help="Number of simulated workers")
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of simulated tasks")
    parser.add_argument("--update-interval", type=int, default=5, help="Interval in seconds for WebSocket real-time monitoring updates")
    parser.add_argument("--generate-anomalies", action="store_true", help="Generate anomalous test results")
    parser.add_argument("--generate-trends", action="store_true", help="Generate performance trends")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--open-browser", action="store_true", help="Open web browser when dashboard is ready")
    
    args = parser.parse_args()
    
    # Handle signals for clean shutdown
    def signal_handler(sig, frame):
        global running
        logger.info(f"Received signal {sig}, shutting down...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the test
    asyncio.run(run_test(args))

if __name__ == "__main__":
    main()