#!/usr/bin/env python3
"""
Example of using the Integrated Analysis System with the Distributed Testing Framework

This script demonstrates how to use the IntegratedAnalysisSystem with the 
Distributed Testing Framework Coordinator to collect, analyze, and visualize test results.

The example showcases:
1. Initializing the IntegratedAnalysisSystem
2. Registering with a coordinator for real-time analysis
3. Processing and analyzing test results
4. Generating comprehensive reports and visualizations
5. Using advanced analytical capabilities like failure pattern detection
6. Working with the circuit breaker analysis features
7. Interactive visualization generation

Usage:
    python result_aggregator_example.py
    
    Optional arguments:
    --no-visualization: Disable visualization features
    --no-ml: Disable machine learning features
    --cleanup: Remove database file after completion
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import coordinator and the integrated analysis system
from coordinator import DistributedTestingCoordinator
from result_aggregator.integrated_analysis_system import IntegratedAnalysisSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Example notification handler
def notification_handler(notification):
    """Example notification handler that prints notifications to the console"""
    print("=" * 80)
    print(f"NOTIFICATION: {notification['type'].upper()} - {notification['severity'].upper()}")
    print(f"Message: {notification['message']}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    if "details" in notification:
        print(f"Score: {notification.get('score', 'N/A')}")
        print(f"Type: {notification.get('anomaly_type', notification.get('type', 'N/A'))}")
    print("=" * 80)

async def run_example():
    """Run the comprehensive example demonstrating all features"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Integrated Analysis System Example')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualization features')
    parser.add_argument('--no-ml', action='store_true', help='Disable machine learning features')
    parser.add_argument('--cleanup', action='store_true', help='Remove database file after completion')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Create a temporary database file
    db_path = "./example_results.duckdb"
    
    print("\n" + "=" * 80)
    print("INTEGRATED ANALYSIS SYSTEM EXAMPLE".center(80))
    print("=" * 80)
    
    # Initialize the coordinator
    print("\n[1] Initializing the Distributed Testing Coordinator...")
    coordinator = DistributedTestingCoordinator(
        db_path=db_path,
        port=8081,
        enable_advanced_scheduler=True,
        enable_plugins=True
    )
    print(f"Coordinator initialized (ID: {coordinator.coordinator_id if hasattr(coordinator, 'coordinator_id') else 'unknown'})")
    
    # Initialize the integrated analysis system
    print("\n[2] Initializing the Integrated Analysis System...")
    analysis_system = IntegratedAnalysisSystem(
        db_path=db_path,
        enable_ml=not args.no_ml,
        enable_visualization=not args.no_visualization,
        enable_real_time_analysis=True,
        analysis_interval=timedelta(minutes=1)  # Shortened for demo
    )
    print(f"Analysis system initialized with:")
    print(f"  - Database path: {db_path}")
    print(f"  - ML enabled: {analysis_system.enable_ml}")
    print(f"  - Visualization enabled: {analysis_system.enable_visualization}")
    print(f"  - Real-time analysis enabled: {analysis_system.enable_real_time_analysis}")
    
    # Register with coordinator
    print("\n[3] Registering with coordinator for real-time analysis...")
    analysis_system.register_with_coordinator(coordinator)
    print("Successfully registered with coordinator")
    
    # Register notification handler
    print("\n[4] Setting up notification handling...")
    analysis_system.register_notification_handler(notification_handler)
    print("Notification handler registered")
    
    # Simulate test results
    print("\n[5] Simulating test execution and result processing...")
    
    # Create workers with different capabilities
    workers = {
        "worker_1": {
            "worker_id": "worker_1",
            "hostname": "host-1.example.com",
            "capabilities": {"hardware": ["cuda", "cpu"]},
            "status": "active",
            "tasks_completed": 0,
            "tasks_failed": 0
        },
        "worker_2": {
            "worker_id": "worker_2",
            "hostname": "host-2.example.com",
            "capabilities": {"hardware": ["cpu"]},
            "status": "active",
            "tasks_completed": 0,
            "tasks_failed": 0
        },
        "worker_3": {
            "worker_id": "worker_3",
            "hostname": "host-3.example.com",
            "capabilities": {"hardware": ["cuda", "cpu"]},
            "status": "active",
            "tasks_completed": 0,
            "tasks_failed": 0
        }
    }
    
    # Add workers to coordinator
    for worker_id, worker_data in workers.items():
        coordinator.workers[worker_id] = worker_data
    
    # Create and execute multiple tasks with different characteristics
    print("  Generating diverse test workload...")
    
    # Define model types for testing
    model_types = ["bert", "vit", "whisper", "t5", "clip"]
    hardware_types = ["cuda", "cpu"]
    batch_sizes = [1, 2, 4, 8, 16]
    
    # Function to create a realistic task result
    def create_task_result(model, hardware, batch_size, status="success", add_anomaly=False):
        # Base metrics
        if hardware == "cuda":
            base_throughput = {"bert": 250, "vit": 180, "whisper": 120, "t5": 150, "clip": 200}
            base_latency = {"bert": 4, "vit": 6, "whisper": 8, "t5": 7, "clip": 5}
            base_memory = {"bert": 1024, "vit": 2048, "whisper": 3072, "t5": 1536, "clip": 2560}
        else:  # CPU
            base_throughput = {"bert": 50, "vit": 35, "whisper": 25, "t5": 30, "clip": 40}
            base_latency = {"bert": 20, "vit": 30, "whisper": 40, "t5": 35, "clip": 25}
            base_memory = {"bert": 1024, "vit": 2048, "whisper": 3072, "t5": 1536, "clip": 2560}
            
        # Add some realistic variation (±10%)
        import random
        variation = lambda x: x * (1 + random.uniform(-0.1, 0.1))
        
        # Scale based on batch size (non-linear)
        throughput = variation(base_throughput[model] * (batch_size ** 0.8)) 
        latency = variation(base_latency[model] * (batch_size ** 0.2))
        memory = variation(base_memory[model] * (batch_size ** 0.5))
        
        # Add circuit breaker data for some results
        circuit_breaker_data = None
        if random.random() < 0.3:  # 30% chance to include circuit breaker data
            states = ["closed", "open", "half_open"]
            circuit_breaker_data = {
                "circuit_breaker_state": random.choice(states),
                "failure_count": random.randint(0, 10),
                "success_streak": random.randint(0, 5),
                "failure_threshold": random.randint(3, 10),
                "recovery_timeout": random.randint(5, 30)
            }
        
        # Create the result object
        result = {
            "status": status,
            "metrics": {
                "throughput": throughput,
                "latency": latency,
                "memory_usage": memory,
                "cpu_usage": random.uniform(10, 90) if hardware == "cpu" else random.uniform(5, 30)
            },
            "details": {
                "model": model,
                "hardware": hardware,
                "batch_size": batch_size,
                "precision": "fp16" if hardware == "cuda" else "fp32",
                "test_duration": random.uniform(5, 15),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add error information for failed tasks
        if status != "success":
            error_types = ["timeout", "out_of_memory", "numerical_error", "driver_error", "connection_error"]
            result["error"] = {
                "type": random.choice(error_types),
                "message": f"Example error message for {random.choice(error_types)}",
                "traceback": "Example traceback information"
            }
            
        # Add circuit breaker data if available
        if circuit_breaker_data:
            result["circuit_breaker"] = circuit_breaker_data
            
        # Add recovery strategy data for some results
        if random.random() < 0.2:  # 20% chance to include recovery data
            strategies = ["retry", "fallback", "circuit_breaker", "timeout_extension", "resource_adjustment"]
            result["recovery_strategy"] = {
                "type": random.choice(strategies),
                "attempts": random.randint(1, 3),
                "success": random.random() < 0.7,  # 70% success rate
                "duration": random.uniform(0.5, 3.0)
            }
            
        # Add anomaly if requested
        if add_anomaly:
            # Create an anomaly by significantly altering a metric
            anomaly_factor = 5 if random.random() < 0.5 else 0.1
            if random.random() < 0.33:
                result["metrics"]["throughput"] *= anomaly_factor
            elif random.random() < 0.5:
                result["metrics"]["latency"] *= anomaly_factor
            else:
                result["metrics"]["memory_usage"] *= anomaly_factor
                
        return result
    
    # Create and process tasks
    task_count = 30  # Create 30 tasks for a good dataset
    print(f"  Generating {task_count} diverse tasks...")
    
    anomaly_tasks = set()  # Track which tasks will have anomalies
    failed_tasks = set()   # Track which tasks will fail
    
    # Select ~10% of tasks to have anomalies and ~15% to fail
    import random
    for i in range(int(task_count * 0.1)):
        anomaly_tasks.add(random.randint(1, task_count))
    for i in range(int(task_count * 0.15)):
        failed_tasks.add(random.randint(1, task_count))
    
    # Process tasks
    for i in range(1, task_count + 1):
        task_id = f"task_{i}"
        # Select random parameters
        model = random.choice(model_types)
        hardware = random.choice(hardware_types)
        batch_size = random.choice(batch_sizes)
        
        # Select a worker based on hardware requirements
        eligible_workers = [w for w, data in workers.items() 
                        if hardware in data["capabilities"]["hardware"]]
        worker_id = random.choice(eligible_workers)
        
        # Determine task status
        status = "failed" if i in failed_tasks else "success"
        
        # Create task in coordinator
        coordinator.tasks[task_id] = {
            "task_id": task_id,
            "type": "benchmark",
            "status": "running",
            "priority": random.randint(1, 3),
            "requirements": {"hardware": [hardware]},
            "metadata": {"model": model, "batch_size": batch_size},
            "attempts": random.randint(1, 3),
            "started": datetime.now().isoformat()
        }
        
        # Associate task with worker
        coordinator.running_tasks[task_id] = worker_id
        
        # Increment worker stats
        workers[worker_id]["tasks_completed"] += 1
        if status == "failed":
            workers[worker_id]["tasks_failed"] += 1
        
        # Create a result with possible anomaly
        result = create_task_result(
            model=model, 
            hardware=hardware, 
            batch_size=batch_size,
            status=status,
            add_anomaly=(i in anomaly_tasks)
        )
        
        # Call the task completed handler
        await coordinator._handle_task_completed(task_id, worker_id, result, 
                                              result["details"]["test_duration"])
        
        # Small delay to make timestamps more realistic
        await asyncio.sleep(0.05)
        
    print(f"  Processed {task_count} tasks with {len(failed_tasks)} failures and {len(anomaly_tasks)} anomalies")
    
    # Wait for real-time analysis to process
    print("\n[6] Waiting for real-time analysis to complete...")
    await asyncio.sleep(2)
    
    # Perform comprehensive analysis
    print("\n[7] Performing comprehensive analysis of results...")
    analysis_results = analysis_system.analyze_results(
        analysis_types=["trends", "anomalies", "workload", "failures", 
                      "performance", "circuit_breaker", "recovery", "forecast"],
        metrics=["throughput", "latency", "memory_usage", "cpu_usage"],
        group_by="hardware"
    )
    
    # Print analysis summary
    print(f"\nAnalysis complete with {len(analysis_results)} result categories")
    print(f"Results include {len(analysis_results.get('trends', {}))} metrics with trend analysis")
    print(f"Detected {len(analysis_results.get('anomalies', []))} anomalies")
    
    # Generate comprehensive report
    print("\n[8] Generating analysis reports in multiple formats...")
    
    # Generate markdown report
    markdown_report = analysis_system.generate_report(
        analysis_results=analysis_results,
        report_type="comprehensive", 
        format="markdown",
        output_path="reports/comprehensive_report.md"
    )
    print(f"Saved markdown report to reports/comprehensive_report.md")
    
    # Generate HTML report
    if analysis_system.enable_visualization:
        html_report = analysis_system.generate_report(
            analysis_results=analysis_results,
            report_type="comprehensive", 
            format="html",
            output_path="reports/comprehensive_report.html"
        )
        print(f"Saved HTML report to reports/comprehensive_report.html")
    
    # Generate JSON report
    json_report = analysis_system.generate_report(
        analysis_results=analysis_results,
        report_type="comprehensive",
        format="json",
        output_path="reports/comprehensive_report.json"
    )
    print(f"Saved JSON report to reports/comprehensive_report.json")
    
    # Generate performance report
    perf_report = analysis_system.generate_report(
        filter_criteria=None,
        report_type="performance",
        format="markdown",
        output_path="reports/performance_report.md"
    )
    print(f"Saved performance report to reports/performance_report.md")
    
    # Print sample of generated report
    print("\nSample of generated report:")
    print("-" * 80)
    # Print first 15 lines of report
    if isinstance(markdown_report, str):
        print("\n".join(markdown_report.split("\n")[:15]))
        print("...")
    print("-" * 80)
    
    # Generate visualizations
    if analysis_system.enable_visualization:
        print("\n[9] Generating visualizations...")
        
        # Generate trend visualization
        trend_success = analysis_system.visualize_results(
            visualization_type="trends",
            data=analysis_results.get("trends"),
            metrics=["throughput", "latency"],
            output_path="visualizations/performance_trends.png"
        )
        if trend_success:
            print(f"Generated performance trend visualization: visualizations/performance_trends.png")
        
        # Generate workload distribution visualization
        workload_success = analysis_system.visualize_results(
            visualization_type="workload_distribution",
            data=analysis_results.get("workload_distribution"),
            output_path="visualizations/workload_distribution.png"
        )
        if workload_success:
            print(f"Generated workload distribution visualization: visualizations/workload_distribution.png")
        
        # Generate anomaly visualization
        anomaly_success = analysis_system.visualize_results(
            visualization_type="anomalies",
            data=analysis_results.get("anomalies"),
            output_path="visualizations/anomalies.png"
        )
        if anomaly_success:
            print(f"Generated anomaly visualization: visualizations/anomalies.png")
        
        # Generate failure patterns visualization
        failure_success = analysis_system.visualize_results(
            visualization_type="failure_patterns",
            data=analysis_results.get("failure_patterns"),
            output_path="visualizations/failure_patterns.png"
        )
        if failure_success:
            print(f"Generated failure patterns visualization: visualizations/failure_patterns.png")
    else:
        print("\n[9] Skipping visualizations (disabled)")
    
    # Demonstrate more advanced features
    print("\n[10] Extracting specific insights...")
    
    # Workload distribution analysis
    if "workload_distribution" in analysis_results:
        workload = analysis_results["workload_distribution"]
        if "distribution_stats" in workload:
            stats = workload["distribution_stats"]
            print("\nWorkload Distribution Analysis:")
            print(f"  Workers: {stats.get('total_workers', 0)}")
            print(f"  Tasks: {stats.get('total_tasks', 0)}")
            print(f"  Mean tasks per worker: {stats.get('mean_tasks_per_worker', 0):.2f}")
            print(f"  Distribution inequality (Gini): {stats.get('gini_coefficient', 0):.2f}")
            
            if "worker_stats" in workload:
                print(f"\n  Worker Performance:")
                for worker_id, worker_stats in workload["worker_stats"].items():
                    print(f"    {worker_id}: {worker_stats.get('total_tasks', 0)} tasks, " +
                         f"{worker_stats.get('success_rate', 0):.1f}% success rate")
    
    # Failure pattern analysis
    if "failure_patterns" in analysis_results:
        failures = analysis_results["failure_patterns"]
        if "failure_counts" in failures:
            print("\nFailure Pattern Analysis:")
            for error_type, count in failures["failure_counts"].items():
                print(f"  {error_type}: {count} occurrences")
            
            if "failure_correlations" in failures:
                print("\n  Key Failure Correlations:")
                for corr in failures["failure_correlations"][:3]:  # Show top 3
                    corr_type = corr.get("type", "unknown")
                    if corr_type == "worker_issue":
                        print(f"    Worker {corr.get('worker_id')}: {corr.get('total_failures')} failures")
                    elif corr_type == "test_type_issue":
                        print(f"    Test type {corr.get('test_type')}: {corr.get('total_failures')} failures")
    
    # Performance forecasting
    if "forecasts" in analysis_results:
        print("\nPerformance Forecasting:")
        for metric, forecast in analysis_results["forecasts"].items():
            if "forecast" in forecast and forecast.get("success", False):
                future_values = forecast["forecast"]
                print(f"  {metric} future trend: {forecast.get('trend_description', 'unknown')}")
                if len(future_values) > 0:
                    print(f"  {metric} next value prediction: {future_values[0]:.2f}")
                if "confidence_intervals" in forecast:
                    ci = forecast["confidence_intervals"]
                    print(f"  95% confidence interval: {ci['lower'][0]:.2f} to {ci['upper'][0]:.2f}")
    
    # Check for circuit breaker analysis
    if "circuit_breaker_performance" in analysis_results:
        circuit_breaker = analysis_results["circuit_breaker_performance"]
        if "transition_stats" in circuit_breaker:
            stats = circuit_breaker["transition_stats"]
            print("\nCircuit Breaker Analysis:")
            if "recovery_success_rate" in stats:
                print(f"  Recovery success rate: {stats['recovery_success_rate']:.1f}%")
            if "transition_counts" in stats:
                print("  Transitions:")
                for transition, count in stats["transition_counts"].items():
                    print(f"    {transition}: {count} occurrences")
    
    # Clean up
    print("\n[11] Cleaning up resources...")
    analysis_system.close()
    
    # Optionally, remove the database file
    if args.cleanup:
        os.remove(db_path)
        print(f"Removed database file: {db_path}")
    
    print("\nExample completed.")
    print("=" * 80)
    print(" RESULTS SUMMARY ".center(80, "="))
    print("=" * 80)
    print(f"✓ Created and processed {task_count} test tasks")
    print(f"✓ Generated comprehensive analysis in multiple metrics")
    print(f"✓ Produced {len(os.listdir('reports'))} analysis reports")
    if analysis_system.enable_visualization:
        print(f"✓ Created {len(os.listdir('visualizations'))} data visualizations")
    print(f"✓ Demonstrated circuit breaker and failure analysis")
    print(f"✓ Showcased performance forecasting capabilities")
    print("=" * 80)
    print("\nAll output saved to the 'reports' and 'visualizations' directories")

if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example())