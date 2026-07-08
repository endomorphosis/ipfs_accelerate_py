#!/usr/bin/env python
"""
API Distributed Testing Runner

This script provides a command-line interface for running distributed tests
on API backends using the APIDistributedTesting framework.

Usage:
    python run_api_distributed_tests.py --api-type openai --test-type latency
    python run_api_distributed_tests.py --api-type groq,claude --test-type throughput --concurrency 5
    python run_api_distributed_tests.py --compare openai,groq,claude --test-type reliability
    python run_api_distributed_tests.py --distributed --coordinator http://localhost:5555 --api-type openai --test-type latency
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API distributed testing framework
try:
    from api_unified_testing_interface import (
        APIDistributedTesting, 
        APITestType,
        APIProvider
    )
except ImportError:
    from test.api_unified_testing_interface import (
        APIDistributedTesting, 
        APITestType,
        APIProvider
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_test_runner")


def setup_argument_parser():
    """Set up and return the argument parser for the command-line interface."""
    parser = argparse.ArgumentParser(description="Run distributed tests on API backends")
    
    # API selection
    api_group = parser.add_mutually_exclusive_group(required=True)
    api_group.add_argument("--api-type", type=str, help="API type(s) to test (comma-separated)")
    api_group.add_argument("--compare", type=str, help="Compare multiple API types (comma-separated)")
    api_group.add_argument("--list", action="store_true", help="List available API types")
    api_group.add_argument("--status", action="store_true", help="Show API framework status")
    
    # Test configuration
    parser.add_argument("--test-type", type=str, choices=[t.value for t in APITestType], 
                      help="Type of test to run")
    parser.add_argument("--prompt", type=str, default="Hello, world!", 
                      help="Prompt to use for testing")
    parser.add_argument("--model", type=str, help="Model to use for testing")
    parser.add_argument("--iterations", type=int, default=5, 
                      help="Number of iterations for latency/reliability tests")
    parser.add_argument("--concurrency", type=int, default=3, 
                      help="Concurrency level for throughput tests")
    parser.add_argument("--duration", type=int, default=10, 
                      help="Duration in seconds for throughput tests")
    
    # Distributed testing options
    parser.add_argument("--distributed", action="store_true", 
                      help="Run tests using the distributed testing framework")
    parser.add_argument("--coordinator", type=str, default="http://localhost:5555", 
                      help="URL of the coordinator server (for distributed tests)")
    parser.add_argument("--priority", type=int, default=10, 
                      help="Priority of the task (for distributed tests)")
    parser.add_argument("--workers", type=int, default=1, 
                      help="Number of workers to use (for distributed tests)")
    parser.add_argument("--worker-tags", type=str,
                      help="Worker tags for task assignment (comma-separated, for distributed tests)")
    parser.add_argument("--async", action="store_true", dest="async_mode",
                      help="Run distributed tests asynchronously and return task IDs")
    
    # Result monitoring
    parser.add_argument("--monitor", type=str, 
                      help="Monitor distributed test by task ID")
    parser.add_argument("--poll-interval", type=int, default=5, 
                      help="Interval in seconds for polling task status (for monitoring)")
    parser.add_argument("--timeout", type=int, default=600, 
                      help="Timeout in seconds for waiting for task completion (for monitoring)")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file for test results (JSON format)")
    parser.add_argument("--anomaly-detection", action="store_true", 
                      help="Run anomaly detection on previous test results")
    parser.add_argument("--lookback-days", type=int, default=7, 
                      help="Days to look back for anomaly detection")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser


def format_test_results(results: Dict[str, Any], test_type: str, api_type: str = None) -> str:
    """Format test results for display."""
    output = []
    
    # Handle API comparison results
    if "results_by_api" in results:
        output.append(f"API Comparison Results - Test: {test_type}")
        output.append("=" * 40)
        
        # Add summary if available
        if "summary" in results:
            if test_type == "latency" and "latency_ranking" in results["summary"]:
                output.append(f"Fastest API: {results['summary']['fastest_api']}")
                output.append(f"Latency Ranking: {', '.join(results['summary']['latency_ranking'])}")
            elif test_type == "throughput" and "throughput_ranking" in results["summary"]:
                output.append(f"Highest Throughput API: {results['summary']['highest_throughput_api']}")
                output.append(f"Throughput Ranking: {', '.join(results['summary']['throughput_ranking'])}")
            elif test_type == "reliability" and "reliability_ranking" in results["summary"]:
                output.append(f"Most Reliable API: {results['summary']['most_reliable_api']}")
                output.append(f"Reliability Ranking: {', '.join(results['summary']['reliability_ranking'])}")
        
        # Add details for each API
        for api, api_results in results["results_by_api"].items():
            output.append(f"\n{api.upper()}")
            output.append("-" * 20)
            
            if api_results.get("status", "") == "error":
                output.append(f"Error: {api_results.get('error', 'Unknown error')}")
                continue
            
            if "summary" in api_results:
                summary = api_results["summary"]
                
                if test_type == "latency":
                    output.append(f"Avg Latency: {summary.get('avg_latency', 'N/A'):.3f}s")
                    output.append(f"Min Latency: {summary.get('min_latency', 'N/A'):.3f}s")
                    output.append(f"Max Latency: {summary.get('max_latency', 'N/A'):.3f}s")
                    output.append(f"Success Rate: {summary.get('successful_iterations', 0)}/{api_results.get('iterations', 0)}")
                
                elif test_type == "throughput":
                    output.append(f"Requests/Second: {summary.get('requests_per_second', 'N/A'):.2f}")
                    output.append(f"Total Requests: {summary.get('total_requests', 'N/A')}")
                    output.append(f"Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                
                elif test_type == "reliability":
                    output.append(f"Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                    output.append(f"Avg Latency: {summary.get('avg_latency', 'N/A'):.3f}s")
                    output.append(f"Successful: {summary.get('successful_iterations', 0)}/{api_results.get('iterations', 0)}")
    
    # Handle single API test results
    else:
        output.append(f"Test Results: {test_type.upper()} - API: {api_type}")
        output.append("=" * 40)
        
        if results.get("status", "") == "error":
            output.append(f"Error: {results.get('error', 'Unknown error')}")
            return "\n".join(output)
        
        if "summary" in results:
            summary = results["summary"]
            
            if test_type == "latency":
                output.append(f"Avg Latency: {summary.get('avg_latency', 'N/A'):.3f}s")
                output.append(f"Min Latency: {summary.get('min_latency', 'N/A'):.3f}s")
                output.append(f"Max Latency: {summary.get('max_latency', 'N/A'):.3f}s")
                output.append(f"Success Rate: {summary.get('successful_iterations', 0)}/{results.get('iterations', 0)}")
            
            elif test_type == "throughput":
                output.append(f"Requests/Second: {summary.get('requests_per_second', 'N/A'):.2f}")
                output.append(f"Total Requests: {summary.get('total_requests', 'N/A')}")
                output.append(f"Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                output.append(f"Duration: {summary.get('actual_duration', 'N/A'):.2f}s")
            
            elif test_type == "reliability":
                output.append(f"Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                output.append(f"Avg Latency: {summary.get('avg_latency', 'N/A'):.3f}s")
                output.append(f"Successful: {summary.get('successful_iterations', 0)}/{results.get('iterations', 0)}")
            
            elif test_type == "cost_efficiency":
                cost_metrics = summary.get("cost_efficiency_metrics", {})
                output.append(f"Tokens per Dollar: {cost_metrics.get('tokens_per_dollar', 'N/A'):.2f}")
                output.append(f"Output Tokens per Dollar: {cost_metrics.get('output_tokens_per_dollar', 'N/A'):.2f}")
                output.append(f"Tokens per Second: {cost_metrics.get('tokens_per_second', 'N/A'):.2f}")
                output.append(f"Output Tokens per Second: {cost_metrics.get('output_tokens_per_second', 'N/A'):.2f}")
                
            elif test_type == "concurrency":
                output.append(f"Optimal Concurrency: {summary.get('optimal_concurrency', 'N/A')}")
                output.append(f"Max Throughput: {summary.get('max_throughput', 'N/A'):.2f} req/s")
                
                # Add details for each concurrency level
                if "concurrency_effects" in summary:
                    output.append("\nConcurrency Effects:")
                    for level, data in summary["concurrency_effects"].items():
                        output.append(f"  Level {level}: {data.get('throughput', 'N/A'):.2f} req/s, " +
                                    f"{data.get('success_rate', 'N/A'):.2%} success rate, " +
                                    f"{data.get('avg_latency', 'N/A'):.3f}s avg latency")
    
    return "\n".join(output)


def format_api_status(status: Dict[str, Any]) -> str:
    """Format API framework status for display."""
    output = []
    
    output.append("API Distributed Testing Framework Status")
    output.append("=" * 40)
    output.append(f"Status: {status['status']}")
    output.append(f"Framework Health: {status['framework_health']['status']}")
    
    output.append("\nAPI Backends:")
    for api_type, api_status in status["api_status"].items():
        output.append(f"  {api_type}: {'Registered' if api_status['registered'] else 'Not Registered'}")
    
    output.append(f"\nAvailable API Types: {', '.join(status['available_api_types'])}")
    output.append(f"Registered API Types: {', '.join(status['registered_api_types'])}")
    
    # Display coordinator information if available
    if "coordinator_status" in status:
        coordinator = status["coordinator_status"]
        output.append("\nCoordinator Status:")
        output.append(f"  Connected: {coordinator['connected']}")
        output.append(f"  URL: {coordinator['url']}")
        output.append(f"  Workers: {coordinator['active_workers']}")
        output.append(f"  Pending Tasks: {coordinator.get('pending_tasks', 'N/A')}")
        output.append(f"  Running Tasks: {coordinator.get('running_tasks', 'N/A')}")
    
    output.append("\nMetrics Summary:")
    metrics = status["metrics_summary"]
    output.append(f"  Test History Count: {metrics['test_history_count']}")
    output.append(f"  APIs with Latency Data: {', '.join(metrics['apis_with_latency_data']) if metrics['apis_with_latency_data'] else 'None'}")
    output.append(f"  APIs with Throughput Data: {', '.join(metrics['apis_with_throughput_data']) if metrics['apis_with_throughput_data'] else 'None'}")
    output.append(f"  APIs with Reliability Data: {', '.join(metrics['apis_with_reliability_data']) if metrics['apis_with_reliability_data'] else 'None'}")
    
    # Format uptime
    uptime = status.get("uptime", 0)
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    seconds = int(uptime % 60)
    output.append(f"\nUptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    return "\n".join(output)


def format_anomaly_detection(results: Dict[str, Any]) -> str:
    """Format anomaly detection results for display."""
    output = []
    
    output.append(f"Anomaly Detection Results - API: {results['api_type']}, Test: {results['test_type']}")
    output.append("=" * 50)
    
    if results.get("status", "") == "error":
        output.append(f"Error: {results.get('error', 'Unknown error')}")
        return "\n".join(output)
    
    if results.get("status", "") == "warning":
        output.append(f"Warning: {results.get('warning', 'Unknown warning')}")
        return "\n".join(output)
    
    output.append(f"Lookback Period: {results['lookback_days']} days")
    output.append(f"Data Points: {results['data_points']}")
    
    if "anomalies" in results:
        output.append("\nDetected Anomalies:")
        for (metric, algorithm), anomaly_data in results["anomalies"].items():
            output.append(f"  {algorithm} detected {len(anomaly_data.get('indices', []))} anomalies")
            if "severity" in anomaly_data:
                output.append(f"  Severity: {anomaly_data['severity']:.2f}")
            
            # Add details for each anomaly
            if "anomalies" in anomaly_data and len(anomaly_data["anomalies"]) > 0:
                output.append("  Top anomalies:")
                for i, anomaly in enumerate(anomaly_data["anomalies"][:5]):  # Show top 5
                    timestamp = datetime.fromtimestamp(anomaly["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    output.append(f"    {i+1}. {timestamp}: {anomaly['value']:.3f} (score: {anomaly['score']:.2f})")
    
    if "forecast" in results:
        output.append("\nForecast:")
        for method, forecast_data in results["forecast"].items():
            if "prediction" in forecast_data:
                output.append(f"  {method} predicts values ranging from " +
                           f"{min(forecast_data['prediction']):.3f} to {max(forecast_data['prediction']):.3f}")
                
                # Add trend direction
                values = forecast_data["prediction"]
                if len(values) > 1:
                    if values[-1] > values[0]:
                        output.append(f"  Trend direction: Increasing ↑")
                    elif values[-1] < values[0]:
                        output.append(f"  Trend direction: Decreasing ↓")
                    else:
                        output.append(f"  Trend direction: Stable →")
    
    return "\n".join(output)


def format_task_status(status: Dict[str, Any]) -> str:
    """Format task status for display."""
    output = []
    
    output.append(f"Task Status: {status['task_id']}")
    output.append("=" * 40)
    output.append(f"Status: {status['status']}")
    
    if status.get('status') == 'COMPLETED':
        output.append(f"Completion Time: {status.get('completion_time', 'Unknown')}")
        
        # Add summary of results
        if "result" in status:
            result = status["result"]
            
            if "summary" in result:
                summary = result["summary"]
                test_type = result.get("test_type", "unknown")
                
                if test_type == "latency":
                    output.append(f"Avg Latency: {summary.get('avg_latency', 'N/A'):.3f}s")
                    output.append(f"Min Latency: {summary.get('min_latency', 'N/A'):.3f}s")
                    output.append(f"Max Latency: {summary.get('max_latency', 'N/A'):.3f}s")
                
                elif test_type == "throughput":
                    output.append(f"Requests/Second: {summary.get('requests_per_second', 'N/A'):.2f}")
                    output.append(f"Total Requests: {summary.get('total_requests', 'N/A')}")
                    output.append(f"Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                
                elif test_type == "reliability":
                    output.append(f"Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                    output.append(f"Successful: {summary.get('successful_iterations', 0)}/{result.get('iterations', 0)}")
                
                elif test_type == "cost_efficiency":
                    cost_metrics = summary.get("cost_efficiency_metrics", {})
                    output.append(f"Tokens per Dollar: {cost_metrics.get('tokens_per_dollar', 'N/A'):.2f}")
    
    elif status.get('status') == 'RUNNING':
        output.append(f"Start Time: {status.get('start_time', 'Unknown')}")
        output.append(f"Worker: {status.get('worker_id', 'Unknown')}")
        output.append(f"Progress: {status.get('progress', 0):.1%}")
    
    elif status.get('status') == 'PENDING':
        output.append(f"Submission Time: {status.get('submission_time', 'Unknown')}")
        output.append(f"Queue Position: {status.get('queue_position', 'Unknown')}")
    
    elif status.get('status') == 'FAILED':
        output.append(f"Failure Time: {status.get('failure_time', 'Unknown')}")
        output.append(f"Error: {status.get('error', 'Unknown error')}")
    
    return "\n".join(output)


def monitor_task(api_testing, task_id, poll_interval, timeout, quiet=False):
    """Monitor the status of a distributed task until completion or timeout."""
    start_time = time.time()
    end_time = start_time + timeout
    
    while time.time() < end_time:
        status = api_testing.get_task_status(task_id)
        
        if not quiet:
            print(f"\rTask {task_id}: {status['status']} ", end="")
            
            if status.get('status') == 'RUNNING' and 'progress' in status:
                print(f"({status['progress']:.1%} complete)", end="")
            
            print("", end="", flush=True)
        
        if status.get('status') in ['COMPLETED', 'FAILED']:
            if not quiet:
                print("\n")  # Add a newline after the progress output
                print(format_task_status(status))
            return status
        
        time.sleep(poll_interval)
    
    if not quiet:
        print(f"\nTimeout after {timeout} seconds waiting for task {task_id}")
    
    return {"task_id": task_id, "status": "TIMEOUT"}


def main():
    """Main entry point for the API distributed testing runner."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize API distributed testing framework
    coordinator_url = args.coordinator if args.distributed else None
    api_testing = APIDistributedTesting(coordinator_url=coordinator_url)
    
    try:
        # Monitor an existing task
        if args.monitor:
            task_id = args.monitor
            status = monitor_task(api_testing, task_id, args.poll_interval, args.timeout, args.quiet)
            
            # Save results to file if requested
            if args.output and status.get('status') == 'COMPLETED':
                output_path = args.output
                
                # Add timestamp to filename if not provided
                if not output_path.endswith(".json"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{output_path}_{timestamp}.json"
                
                with open(output_path, "w") as f:
                    json.dump(status.get('result', {}), f, indent=2)
                
                if not args.quiet:
                    print(f"\nResults saved to: {output_path}")
            
            return
        
        # List available API types
        if args.list:
            api_info = api_testing.get_api_backend_info()
            print(f"Available API Types:")
            for api_type in api_info["available_api_types"]:
                print(f"  {api_type}")
            return
        
        # Show API framework status
        if args.status:
            framework_status = api_testing.get_api_framework_status()
            print(format_api_status(framework_status))
            return
        
        # Validate test type is provided when needed
        if args.api_type or args.compare:
            if not args.test_type:
                print("Error: --test-type is required when testing APIs")
                parser.print_help()
                return
            
            # Parse API types
            if args.api_type:
                api_types = [api.strip() for api in args.api_type.split(",")]
            elif args.compare:
                api_types = [api.strip() for api in args.compare.split(",")]
            else:
                api_types = []
            
            # Prepare messages for chat completion
            messages = [{"role": "user", "content": args.prompt}]
            
            # Validate test parameters
            test_parameters = {
                "messages": messages,
                "model": args.model
            }
            
            # Add test-specific parameters
            if args.test_type == "latency":
                test_parameters["iterations"] = args.iterations
            elif args.test_type == "throughput":
                test_parameters["concurrent_requests"] = args.concurrency
                test_parameters["duration"] = args.duration
            elif args.test_type == "reliability":
                test_parameters["iterations"] = args.iterations
            elif args.test_type == "cost_efficiency":
                test_parameters["iterations"] = args.iterations
            elif args.test_type == "concurrency":
                test_parameters["concurrency_levels"] = [1, 2, 3, 5, 10][:args.concurrency]
                test_parameters["requests_per_level"] = args.iterations
            
            # Parse worker tags if provided
            worker_tags = []
            if args.worker_tags:
                worker_tags = [tag.strip() for tag in args.worker_tags.split(",")]
            
            # Run tests
            if args.compare:
                # Compare multiple APIs
                if not args.quiet:
                    print(f"Comparing APIs: {', '.join(api_types)} - Test: {args.test_type}")
                
                if args.distributed:
                    # Distributed comparison
                    comparison_id = api_testing.compare_apis(
                        api_types=api_types,
                        test_type=args.test_type,
                        parameters=test_parameters,
                        num_workers_per_api=args.workers
                    )
                    
                    if args.async_mode:
                        print(f"Distributed comparison started with ID: {comparison_id}")
                        print(f"Monitor results with: --monitor {comparison_id}")
                        return
                    
                    # Wait for results
                    if not args.quiet:
                        print(f"Waiting for comparison results...")
                    
                    results = api_testing.get_comparison_results(comparison_id)
                    
                    # Poll until all results are available
                    while any(result.get('status') == 'pending' for result in results.get('results', {}).values()):
                        time.sleep(args.poll_interval)
                        results = api_testing.get_comparison_results(comparison_id)
                else:
                    # Local comparison
                    results = api_testing.compare_apis(
                        api_types=api_types,
                        test_type=args.test_type,
                        parameters=test_parameters
                    )
                
            else:
                # Test a single API
                api_type = api_types[0]
                if not args.quiet:
                    print(f"Testing API: {api_type} - Test: {args.test_type}")
                
                if args.distributed:
                    # Distributed test
                    task_id = api_testing.run_distributed_test(
                        api_type=api_type,
                        test_type=args.test_type,
                        parameters=test_parameters,
                        num_workers=args.workers,
                        worker_tags=worker_tags,
                        priority=args.priority
                    )
                    
                    if args.async_mode:
                        print(f"Distributed test started with ID: {task_id}")
                        print(f"Monitor results with: --monitor {task_id}")
                        return
                    
                    # Wait for results
                    if not args.quiet:
                        print(f"Waiting for test results...")
                    
                    status = monitor_task(api_testing, task_id, args.poll_interval, args.timeout, args.quiet)
                    if status.get('status') == 'COMPLETED':
                        results = status.get('result', {})
                    else:
                        print(f"Test did not complete successfully. Status: {status.get('status')}")
                        return
                else:
                    # Local test
                    results = api_testing.run_test(
                        api_type=api_type,
                        test_type=args.test_type,
                        parameters=test_parameters
                    )
            
            # Run anomaly detection if requested
            if args.anomaly_detection:
                for api_type in api_types:
                    if not args.quiet:
                        print(f"\nRunning anomaly detection for {api_type} - {args.test_type}...")
                    
                    anomaly_results = api_testing.detect_anomalies(
                        api_type=api_type,
                        test_type=args.test_type,
                        lookback_days=args.lookback_days
                    )
                    
                    if not args.quiet:
                        print(format_anomaly_detection(anomaly_results))
                    
                    # Add anomaly results to the main results
                    if "anomalies" not in results:
                        results["anomalies"] = {}
                    
                    results["anomalies"][api_type] = anomaly_results
            
            # Format and display results
            if not args.quiet:
                if args.compare:
                    print(format_test_results(results, args.test_type))
                else:
                    print(format_test_results(results, args.test_type, api_type))
            
            # Save results to file if requested
            if args.output:
                output_path = args.output
                
                # Add timestamp to filename if not provided
                if not output_path.endswith(".json"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{output_path}_{timestamp}.json"
                
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                
                if not args.quiet:
                    print(f"\nResults saved to: {output_path}")
    
    finally:
        # Stop the framework
        api_testing.stop()


if __name__ == "__main__":
    main()