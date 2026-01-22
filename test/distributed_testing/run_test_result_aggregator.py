#!/usr/bin/env python3
"""
Run Result Aggregator Integration Test

This script demonstrates and tests the Result Aggregator System integration with the
Distributed Testing Framework Coordinator. It simulates a complete workflow with task
creation, execution, result processing, and analysis.

Usage:
    python run_test_result_aggregator.py [--db-path DB_PATH] [--enable-ml] [--enable-visualization]
    
    --db-path: Path to DuckDB database (default: ./test_results.duckdb)
    --enable-ml: Enable machine learning for anomaly detection (default: True)
    --enable-visualization: Enable visualization (default: True)
    --quick-test: Run a quick test with minimal tasks
    --generate-anomalies: Generate anomalous test results
    --output-dir: Directory to store generated reports (default: ./reports)
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_test_result_aggregator")

# Import coordinator and result aggregator
from coordinator import DistributedTestingCoordinator
from result_aggregator.coordinator_integration import ResultAggregatorIntegration
from result_aggregator.service import ResultAggregatorService

# Task types for simulation
TASK_TYPES = [
    "benchmark",
    "unit_test",
    "integration_test",
    "functional_test",
    "performance_test"
]

# Hardware types for simulation
HARDWARE_TYPES = [
    "cpu",
    "cuda",
    "rocm",
    "mps",
    "webgpu",
    "webnn"
]

# Notification callback
def notification_callback(notification):
    """Handle notifications from the Result Aggregator"""
    print("\n" + "=" * 80)
    print(f"NOTIFICATION: {notification['type'].upper()} - {notification['severity'].upper()}")
    print(f"Message: {notification['message']}")
    print(f"Timestamp: {notification['timestamp']}")
    
    if "details" in notification and notification["details"]:
        print("\nDetails:")
        if notification["type"] == "anomaly":
            if "anomalous_features" in notification["details"]:
                for feature in notification["details"]["anomalous_features"]:
                    print(f"  • {feature['feature']}: value={feature['value']:.2f}, z-score={feature['z_score']:.2f}")
        elif notification["type"] == "trend":
            print(f"  • Metric: {notification['details']['metric']}")
            print(f"  • Trend: {notification['details']['trend']}")
            print(f"  • Percent Change: {notification['details']['percent_change']:.2f}%")
    
    print("=" * 80)

def generate_random_task(task_id, task_type, hardware_requirements=None, priority=None):
    """Generate a random task with specified parameters"""
    if hardware_requirements is None:
        # Random hardware requirements
        hardware_count = random.randint(1, 3)
        hardware_requirements = random.sample(HARDWARE_TYPES, hardware_count)
    
    if priority is None:
        # Random priority between 1 and 5
        priority = random.randint(1, 5)
    
    return {
        "task_id": task_id,
        "type": task_type,
        "status": "pending",
        "priority": priority,
        "requirements": {"hardware": hardware_requirements},
        "metadata": {
            "model": f"example_model_{random.randint(1, 10)}",
            "batch_size": random.choice([1, 2, 4, 8, 16, 32]),
            "precision": random.choice(["fp32", "fp16", "int8", "int4"])
        },
        "attempts": 0,
        "created": datetime.now().isoformat()
    }

def generate_random_worker(worker_id, hardware_capabilities=None, hostname=None):
    """Generate a random worker with specified parameters"""
    if hardware_capabilities is None:
        # Random hardware capabilities
        hardware_count = random.randint(1, 4)
        hardware_capabilities = random.sample(HARDWARE_TYPES, hardware_count)
    
    if hostname is None:
        hostname = f"worker-{random.randint(1, 100)}.example.com"
    
    return {
        "worker_id": worker_id,
        "hostname": hostname,
        "capabilities": {"hardware": hardware_capabilities},
        "status": "active",
        "tasks_completed": 0,
        "tasks_failed": 0,
        "registered": datetime.now().isoformat(),
        "last_heartbeat": datetime.now().isoformat()
    }

def generate_task_result(task, worker, success=True, execution_time=None, generate_anomaly=False):
    """Generate a random task result"""
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

async def run_simulation(args):
    """Run the result aggregator integration simulation"""
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Print simulation parameters
    print("\n" + "=" * 80)
    print("RESULT AGGREGATOR INTEGRATION TEST")
    print("=" * 80)
    print(f"Database Path: {args.db_path}")
    print(f"ML Enabled: {args.enable_ml}")
    print(f"Visualization Enabled: {args.enable_visualization}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Generate Anomalies: {args.generate_anomalies}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80 + "\n")
    
    # Initialize the coordinator
    print("Initializing coordinator...")
    coordinator = DistributedTestingCoordinator(
        db_path=args.db_path,
        port=8081,
        enable_advanced_scheduler=True,
        enable_plugins=True
    )
    
    # Initialize the result aggregator integration
    print("Initializing result aggregator integration...")
    integration = ResultAggregatorIntegration(
        coordinator=coordinator,
        enable_ml=args.enable_ml,
        enable_visualization=args.enable_visualization,
        enable_real_time_analysis=True,
        enable_notifications=True
    )
    
    # Register with coordinator
    integration.register_with_coordinator()
    
    # Register notification callback
    integration.register_notification_callback(notification_callback)
    
    # Determine number of tasks and workers based on test type
    if args.quick_test:
        num_workers = 3
        num_tasks = 10
    else:
        num_workers = 10
        num_tasks = 50
    
    # Create workers
    print(f"Creating {num_workers} workers...")
    for i in range(num_workers):
        worker_id = f"worker-{i+1}"
        worker = generate_random_worker(worker_id)
        coordinator.workers[worker_id] = worker
        print(f"  Created worker {worker_id} with hardware: {worker['capabilities']['hardware']}")
    
    # Create tasks
    print(f"Creating {num_tasks} tasks...")
    tasks = []
    for i in range(num_tasks):
        task_id = f"task-{i+1}"
        task_type = random.choice(TASK_TYPES)
        task = generate_random_task(task_id, task_type)
        coordinator.tasks[task_id] = task
        tasks.append(task)
        print(f"  Created {task_type} task {task_id} with requirements: {task['requirements']}")
    
    # Simulate task execution and result processing
    print("\nSimulating task execution...")
    completed_count = 0
    failed_count = 0
    
    for task in tasks:
        task_id = task["task_id"]
        
        # Find compatible worker
        compatible_workers = []
        for worker_id, worker in coordinator.workers.items():
            # Check if worker has all required hardware
            required_hardware = task["requirements"]["hardware"]
            worker_hardware = worker["capabilities"]["hardware"]
            
            if all(hw in worker_hardware for hw in required_hardware):
                compatible_workers.append((worker_id, worker))
        
        if not compatible_workers:
            print(f"  No compatible worker found for task {task_id}")
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
        await asyncio.sleep(0.1)
        
        # Determine success or failure
        success = random.random() < 0.9  # 90% success rate
        
        # Generate result
        result, execution_time = generate_task_result(
            task, 
            worker, 
            success=success,
            generate_anomaly=args.generate_anomalies
        )
        
        if success:
            # Call task completion handler
            await coordinator._handle_task_completed(task_id, worker_id, result, execution_time)
            completed_count += 1
            
            print(f"  Task {task_id} completed by worker {worker_id} in {execution_time:.2f}s")
        else:
            # Call task failure handler
            await coordinator._handle_task_failed(task_id, worker_id, result["error"], execution_time)
            failed_count += 1
            
            print(f"  Task {task_id} failed on worker {worker_id}: {result['error']}")
    
    print(f"\nTask execution simulation complete: {completed_count} completed, {failed_count} failed")
    
    # Wait for real-time analysis to complete
    print("\nWaiting for real-time analysis to complete...")
    await asyncio.sleep(2)
    
    # Generate a trend simulation by creating tasks with gradually changing metrics
    if not args.quick_test:
        print("\nSimulating performance trend...")
        
        # Create a new task type for trending
        task_type = "benchmark"
        worker_id = list(coordinator.workers.keys())[0]
        worker = coordinator.workers[worker_id]
        
        # Generate tasks with trending metrics
        trend_tasks = []
        num_trend_tasks = 20
        
        for i in range(num_trend_tasks):
            task_id = f"trend-task-{i+1}"
            task = generate_random_task(
                task_id,
                task_type,
                hardware_requirements=["cuda"],
                priority=3
            )
            
            # Use consistent metadata for trending
            task["metadata"] = {
                "model": "trending_model",
                "batch_size": 8,
                "precision": "fp16"
            }
            
            coordinator.tasks[task_id] = task
            trend_tasks.append(task)
        
        # Simulate execution with trending metrics
        print(f"  Executing {num_trend_tasks} trending tasks...")
        
        base_throughput = 100.0
        base_latency = 10.0
        base_memory = 1000.0
        base_execution_time = 10.0
        
        for i, task in enumerate(trend_tasks):
            task_id = task["task_id"]
            
            # Associate task with worker
            coordinator.running_tasks[task_id] = worker_id
            
            # Update task status
            task["status"] = "running"
            task["started"] = datetime.now().isoformat()
            task["attempts"] += 1
            
            # Calculate trend factor (gradually increasing or decreasing)
            trend_factor = 1.0 + (i / num_trend_tasks) * 0.5  # 50% increase over the trend
            
            # Create result with trending metrics
            execution_time = base_execution_time * (trend_factor ** 0.5)
            
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
            
            # Call task completion handler
            await coordinator._handle_task_completed(task_id, worker_id, result, execution_time)
            
            # Add a small delay between tasks
            await asyncio.sleep(0.1)
        
        print("  Trending tasks executed successfully")
    
    # Wait for trend analysis to complete
    print("\nWaiting for trend analysis to complete...")
    await asyncio.sleep(2)
    
    # Generate performance reports
    print("\nGenerating performance report...")
    
    # Generate summary report
    summary_report = integration.service.generate_analysis_report(
        report_type="summary",
        format="markdown"
    )
    
    if args.output_dir:
        summary_report_path = os.path.join(args.output_dir, "summary_report.md")
        with open(summary_report_path, "w") as f:
            f.write(summary_report)
        print(f"  Summary report saved to {summary_report_path}")
    
    # Generate performance report
    performance_report = integration.service.generate_analysis_report(
        report_type="performance",
        format="markdown"
    )
    
    if args.output_dir:
        performance_report_path = os.path.join(args.output_dir, "performance_report.md")
        with open(performance_report_path, "w") as f:
            f.write(performance_report)
        print(f"  Performance report saved to {performance_report_path}")
    
    # Generate anomaly report if anomalies were generated
    if args.generate_anomalies:
        anomaly_report = integration.service.generate_analysis_report(
            report_type="anomaly",
            format="markdown"
        )
        
        if args.output_dir:
            anomaly_report_path = os.path.join(args.output_dir, "anomaly_report.md")
            with open(anomaly_report_path, "w") as f:
                f.write(anomaly_report)
            print(f"  Anomaly report saved to {anomaly_report_path}")
    
    # Print trend analysis
    print("\nAnalyzing performance trends...")
    trends = integration.service.analyze_performance_trends()
    
    significant_trends = []
    for metric, trend_data in trends.items():
        if "trend" in trend_data and trend_data["trend"] != "stable" and abs(trend_data.get("percent_change", 0)) > 5:
            significant_trends.append((metric, trend_data))
    
    if significant_trends:
        print(f"  Found {len(significant_trends)} significant trends:")
        for metric, trend_data in significant_trends:
            trend = trend_data.get("trend", "unknown")
            percent_change = trend_data.get("percent_change", 0)
            print(f"  • {metric}: {trend} by {abs(percent_change):.2f}%")
    else:
        print("  No significant trends detected")
    
    # Print anomaly detection
    if args.generate_anomalies:
        print("\nDetecting anomalies...")
        anomalies = integration.service.detect_anomalies()
        
        if anomalies:
            print(f"  Detected {len(anomalies)} anomalies:")
            for i, anomaly in enumerate(anomalies[:5]):  # Show top 5
                score = anomaly.get("score", 0)
                anomaly_type = anomaly.get("type", "unknown")
                print(f"  • Anomaly {i+1}: {anomaly_type} with score {score:.4f}")
                
                if "anomalous_features" in anomaly.get("details", {}):
                    for feature in anomaly["details"]["anomalous_features"][:3]:  # Show top 3 features
                        feature_name = feature.get("feature", "unknown")
                        value = feature.get("value", 0)
                        z_score = feature.get("z_score", 0)
                        print(f"    - {feature_name}: value={value:.2f}, z-score={z_score:.2f}")
            
            if len(anomalies) > 5:
                print(f"    ... and {len(anomalies) - 5} more")
        else:
            print("  No anomalies detected")
    
    # Display summary metrics
    if args.output_dir:
        print(f"\nAll reports saved to {args.output_dir}")
    
    # Final summary
    print("\nResult Aggregator Integration Test completed successfully!")
    print(f"  Tasks: {num_tasks} total, {completed_count} completed, {failed_count} failed")
    print(f"  Workers: {num_workers}")
    
    if significant_trends:
        print(f"  Trends: {len(significant_trends)} significant trends detected")
    
    if args.generate_anomalies and anomalies:
        print(f"  Anomalies: {len(anomalies)} anomalies detected")
    
    print("\nCleanup...")
    integration.close()
    print("Done!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Result Aggregator Integration Test")
    parser.add_argument("--db-path", default="./test_results.duckdb", help="Path to DuckDB database")
    parser.add_argument("--enable-ml", action="store_true", default=True, help="Enable machine learning")
    parser.add_argument("--enable-visualization", action="store_true", default=True, help="Enable visualization")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with minimal tasks")
    parser.add_argument("--generate-anomalies", action="store_true", help="Generate anomalous test results")
    parser.add_argument("--output-dir", default="./reports", help="Directory to store generated reports")
    
    args = parser.parse_args()
    
    # Run the simulation
    asyncio.run(run_simulation(args))

if __name__ == "__main__":
    main()