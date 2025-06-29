"""
End-to-End Testing for Distributed Testing Framework

This script provides comprehensive end-to-end testing for the Distributed Testing Framework,
focusing on the integration between the Monitoring Dashboard and Result Aggregator.

Usage:
    python -m duckdb_api.distributed_testing.tests.test_end_to_end_framework

Options:
    --workers INT                 Number of simulated worker nodes (default: 5)
    --test-duration INT           Duration of tests in seconds (default: 60)
    --hardware-profiles STR       Hardware profiles to simulate [cpu,gpu,webgpu,webnn,multi] (default: all)
    --include-failures            Include simulated failures
    --dashboard-port INT          Port for the monitoring dashboard (default: 8080)
    --coordinator-port INT        Port for the coordinator service (default: 8081)
    --result-aggregator-port INT  Port for the result aggregator service (default: 8082)
    --cache-dir STR               Directory for caching test data (default: ./.e2e_test_cache)
    --report-dir STR              Directory for test reports (default: ./e2e_test_reports)
    --debug                       Enable debug logging
    --headless                    Run in headless mode without browser interaction
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to ensure imports work properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from duckdb_api.distributed_testing.coordinator.coordinator_service import CoordinatorService
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_result_aggregator_integration import ResultAggregatorIntegration
from duckdb_api.distributed_testing.fault_tolerance.fault_tolerance_manager import FaultToleranceManager
from duckdb_api.distributed_testing.load_balancer.load_balancer_service import LoadBalancerService
from duckdb_api.distributed_testing.result_aggregator.result_aggregator_service import ResultAggregatorService
from duckdb_api.distributed_testing.worker.worker_node import WorkerNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("e2e_test_framework.log")
    ]
)
logger = logging.getLogger("E2ETest")

# Constants
TEST_MODELS = [
    "bert-base-uncased",
    "t5-small",
    "vit-base-patch16-224",
    "whisper-tiny",
    "llama-7b"
]
HARDWARE_TYPES = ["cpu", "gpu", "webgpu", "webnn", "multi"]
TEST_TYPES = ["performance", "compatibility", "integration", "web_platform"]

class EndToEndTestFramework:
    """Comprehensive end-to-end test framework for the Distributed Testing Framework."""
    
    def __init__(
        self,
        num_workers: int = 5,
        test_duration: int = 60,
        hardware_profiles: List[str] = None,
        include_failures: bool = False,
        dashboard_port: int = 8080,
        coordinator_port: int = 8081,
        result_aggregator_port: int = 8082,
        cache_dir: str = "./.e2e_test_cache",
        report_dir: str = "./e2e_test_reports",
        debug: bool = False,
        headless: bool = False
    ):
        """Initialize the end-to-end test framework.
        
        Args:
            num_workers: Number of simulated worker nodes
            test_duration: Duration of the test in seconds
            hardware_profiles: Hardware profiles to simulate
            include_failures: Whether to include simulated failures
            dashboard_port: Port for the monitoring dashboard
            coordinator_port: Port for the coordinator service
            result_aggregator_port: Port for the result aggregator service
            cache_dir: Directory for caching test data
            report_dir: Directory for test reports
            debug: Enable debug logging
            headless: Run in headless mode without browser interaction
        """
        # Set up configuration
        self.num_workers = num_workers
        self.test_duration = test_duration
        self.hardware_profiles = hardware_profiles or HARDWARE_TYPES
        self.include_failures = include_failures
        self.dashboard_port = dashboard_port
        self.coordinator_port = coordinator_port
        self.result_aggregator_port = result_aggregator_port
        self.cache_dir = Path(cache_dir)
        self.report_dir = Path(report_dir)
        self.debug = debug
        self.headless = headless
        
        # Configure logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processes = []
        self.workers = []
        self.worker_processes = []
        self.test_id = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_file = self.report_dir / f"{self.test_id}_results.json"
        self.validation_results = {}
        
        logger.info(f"Initialized E2E Test Framework with ID: {self.test_id}")
        logger.info(f"Configuration: {num_workers} workers, {test_duration}s duration")
        logger.info(f"Hardware profiles: {', '.join(self.hardware_profiles)}")
        logger.info(f"Ports: Dashboard={dashboard_port}, Coordinator={coordinator_port}, ResultAggregator={result_aggregator_port}")
    
    async def start_services(self):
        """Start all required services for testing."""
        logger.info("Starting services...")
        
        # Start the result aggregator service
        result_aggregator_cmd = [
            "python", "-m", "duckdb_api.distributed_testing.result_aggregator.run_result_aggregator",
            "--port", str(self.result_aggregator_port),
            "--db-path", str(self.cache_dir / "test_results.duckdb")
        ]
        if self.debug:
            result_aggregator_cmd.append("--debug")
        
        result_aggregator_process = subprocess.Popen(
            result_aggregator_cmd,
            stdout=subprocess.PIPE if not self.debug else None,
            stderr=subprocess.PIPE if not self.debug else None
        )
        self.processes.append(result_aggregator_process)
        logger.info(f"Started Result Aggregator Service on port {self.result_aggregator_port}")
        
        # Start the coordinator service
        coordinator_cmd = [
            "python", "-m", "duckdb_api.distributed_testing.coordinator.run_coordinator",
            "--port", str(self.coordinator_port),
            "--result-aggregator-url", f"http://localhost:{self.result_aggregator_port}"
        ]
        if self.debug:
            coordinator_cmd.append("--debug")
        
        coordinator_process = subprocess.Popen(
            coordinator_cmd,
            stdout=subprocess.PIPE if not self.debug else None,
            stderr=subprocess.PIPE if not self.debug else None
        )
        self.processes.append(coordinator_process)
        logger.info(f"Started Coordinator Service on port {self.coordinator_port}")
        
        # Start the monitoring dashboard
        dashboard_cmd = [
            "python", "-m", "duckdb_api.distributed_testing.run_monitoring_dashboard",
            "--port", str(self.dashboard_port),
            "--coordinator-url", f"http://localhost:{self.coordinator_port}",
            "--result-aggregator-url", f"http://localhost:{self.result_aggregator_port}",
            "--enable-result-aggregator-integration"
        ]
        if self.debug:
            dashboard_cmd.append("--debug")
        
        dashboard_process = subprocess.Popen(
            dashboard_cmd,
            stdout=subprocess.PIPE if not self.debug else None,
            stderr=subprocess.PIPE if not self.debug else None
        )
        self.processes.append(dashboard_process)
        logger.info(f"Started Monitoring Dashboard on port {self.dashboard_port}")
        
        # Wait for services to start
        await asyncio.sleep(5)
        logger.info("All services started successfully")
    
    async def start_workers(self):
        """Start simulated worker nodes."""
        logger.info(f"Starting {self.num_workers} worker nodes...")
        
        for i in range(self.num_workers):
            # Determine hardware profile
            if len(self.hardware_profiles) == 1:
                hw_profile = self.hardware_profiles[0]
            else:
                hw_profile = random.choice(self.hardware_profiles)
            
            # Configure worker
            worker_cmd = [
                "python", "-m", "duckdb_api.distributed_testing.worker.run_worker",
                "--worker-id", f"worker_{i}_{hw_profile}",
                "--coordinator-url", f"http://localhost:{self.coordinator_port}",
                "--hardware-profile", hw_profile,
                "--simulate-hardware"
            ]
            if self.debug:
                worker_cmd.append("--debug")
            
            # Start worker process
            worker_process = subprocess.Popen(
                worker_cmd,
                stdout=subprocess.PIPE if not self.debug else None,
                stderr=subprocess.PIPE if not self.debug else None
            )
            self.worker_processes.append(worker_process)
            logger.info(f"Started Worker {i} with {hw_profile} hardware profile")
            
            # Stagger worker startup
            await asyncio.sleep(0.5)
        
        # Wait for workers to register
        await asyncio.sleep(5)
        logger.info(f"Started {self.num_workers} worker nodes")
    
    async def submit_test_workloads(self):
        """Submit various test workloads to the coordinator."""
        logger.info("Submitting test workloads...")
        
        async with aiohttp.ClientSession() as session:
            # Create various test types
            for test_type in TEST_TYPES:
                for model in TEST_MODELS:
                    # Create between 2-5 tests for each combination
                    num_tests = random.randint(2, 5)
                    for i in range(num_tests):
                        # Create test configuration
                        test_config = {
                            "test_id": f"{test_type}_{model.replace('-', '_')}_{i}",
                            "test_type": test_type,
                            "model_id": model,
                            "parameters": {
                                "batch_size": random.choice([1, 2, 4, 8, 16]),
                                "precision": random.choice(["fp32", "fp16", "int8", "int4"]),
                                "iterations": random.randint(10, 100)
                            },
                            "hardware_requirements": {
                                "preferred_type": random.choice(self.hardware_profiles),
                                "min_memory_gb": random.choice([2, 4, 8, 16]),
                                "min_compute_capability": random.uniform(5.0, 8.0)
                            },
                            "priority": random.randint(1, 5)
                        }
                        
                        # Submit test
                        async with session.post(
                            f"http://localhost:{self.coordinator_port}/api/tests/submit",
                            json=test_config
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                logger.debug(f"Submitted test {test_config['test_id']}: {result}")
                            else:
                                logger.error(f"Failed to submit test {test_config['test_id']}: {await response.text()}")
                    
                    # Add some sleep to prevent overwhelming the system
                    await asyncio.sleep(0.1)
        
        logger.info("Submitted all test workloads")
    
    async def inject_failures(self):
        """Simulate random failures in the system."""
        if not self.include_failures:
            return
        
        logger.info("Injecting random failures...")
        
        # Wait a bit before injecting failures
        await asyncio.sleep(self.test_duration // 4)
        
        # Kill a random worker
        if self.worker_processes:
            worker_to_kill = random.choice(self.worker_processes)
            try:
                worker_to_kill.terminate()
                logger.info("Terminated a random worker to simulate failure")
            except Exception as e:
                logger.error(f"Failed to terminate worker: {e}")
        
        # Wait a bit more
        await asyncio.sleep(self.test_duration // 4)
        
        # Simulate network issues by sending invalid requests
        async with aiohttp.ClientSession() as session:
            try:
                # Send malformed request to coordinator
                await session.post(
                    f"http://localhost:{self.coordinator_port}/api/tests/submit",
                    json={"invalid": "data"}
                )
                logger.info("Sent malformed request to coordinator")
                
                # Send malformed request to result aggregator
                await session.post(
                    f"http://localhost:{self.result_aggregator_port}/api/results/submit",
                    json={"invalid": "data"}
                )
                logger.info("Sent malformed request to result aggregator")
            except Exception as e:
                logger.debug(f"Exception during failure injection (expected): {e}")
    
    async def wait_for_completion(self):
        """Wait for test execution to complete."""
        logger.info(f"Waiting for {self.test_duration} seconds for test execution...")
        
        # Create progress bar for test duration
        with tqdm(total=self.test_duration, desc="Test Execution", unit="s") as pbar:
            for _ in range(self.test_duration):
                await asyncio.sleep(1)
                pbar.update(1)
        
        logger.info("Test execution period completed")
    
    async def validate_dashboard(self):
        """Validate that the monitoring dashboard is correctly displaying data."""
        logger.info("Validating monitoring dashboard...")
        
        async with aiohttp.ClientSession() as session:
            # Check main dashboard page
            async with session.get(f"http://localhost:{self.dashboard_port}/") as response:
                if response.status == 200:
                    html = await response.text()
                    # Basic validation that dashboard is working
                    validation_results = {
                        "dashboard_accessible": True,
                        "dashboard_content_length": len(html),
                        "coordinator_status_shown": "Coordinator Status" in html,
                        "worker_status_shown": "Worker Status" in html
                    }
                    logger.info(f"Dashboard validation: {json.dumps(validation_results, indent=2)}")
                    self.validation_results["dashboard"] = validation_results
                else:
                    logger.error(f"Failed to access dashboard: {response.status}")
                    self.validation_results["dashboard"] = {"dashboard_accessible": False}
            
            # Check results page
            async with session.get(f"http://localhost:{self.dashboard_port}/results") as response:
                if response.status == 200:
                    html = await response.text()
                    # Validate results page
                    validation_results = {
                        "results_page_accessible": True,
                        "results_content_length": len(html),
                        "performance_tab_shown": "Performance Trends" in html,
                        "compatibility_tab_shown": "Compatibility Matrix" in html,
                        "integration_tab_shown": "Integration Tests" in html,
                        "web_platform_tab_shown": "Web Platform Tests" in html
                    }
                    logger.info(f"Results page validation: {json.dumps(validation_results, indent=2)}")
                    self.validation_results["results_page"] = validation_results
                else:
                    logger.error(f"Failed to access results page: {response.status}")
                    self.validation_results["results_page"] = {"results_page_accessible": False}
    
    async def validate_results(self):
        """Validate that test results are being properly aggregated."""
        logger.info("Validating result aggregation...")
        
        async with aiohttp.ClientSession() as session:
            # Check result aggregator API
            async with session.get(f"http://localhost:{self.result_aggregator_port}/api/results/summary") as response:
                if response.status == 200:
                    summary = await response.json()
                    # Validate result summary
                    validation_results = {
                        "summary_accessible": True,
                        "has_test_results": len(summary.get("test_count", 0)) > 0,
                        "test_types_covered": summary.get("test_types", []),
                        "models_covered": summary.get("models", [])
                    }
                    logger.info(f"Result aggregation validation: {json.dumps(validation_results, indent=2)}")
                    self.validation_results["result_aggregation"] = validation_results
                else:
                    logger.error(f"Failed to access result summary: {response.status}")
                    self.validation_results["result_aggregation"] = {"summary_accessible": False}
            
            # Check detailed results
            for test_type in TEST_TYPES:
                async with session.get(
                    f"http://localhost:{self.result_aggregator_port}/api/results/by-type/{test_type}"
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        self.validation_results[f"results_{test_type}"] = {
                            "accessible": True,
                            "result_count": len(results),
                            "has_data": len(results) > 0
                        }
                    else:
                        logger.error(f"Failed to access {test_type} results: {response.status}")
                        self.validation_results[f"results_{test_type}"] = {"accessible": False}
    
    async def validate_integration(self):
        """Validate the integration between the dashboard and result aggregator."""
        logger.info("Validating dashboard and result aggregator integration...")
        
        async with aiohttp.ClientSession() as session:
            # Check integrated visualization data
            async with session.get(
                f"http://localhost:{self.dashboard_port}/api/results/visualization-data?type=performance"
            ) as response:
                if response.status == 200:
                    viz_data = await response.json()
                    validation_results = {
                        "visualization_data_accessible": True,
                        "has_data": bool(viz_data),
                        "data_types": list(viz_data.keys()) if isinstance(viz_data, dict) else []
                    }
                    logger.info(f"Integration validation: {json.dumps(validation_results, indent=2)}")
                    self.validation_results["integration"] = validation_results
                else:
                    logger.error(f"Failed to access visualization data: {response.status}")
                    self.validation_results["integration"] = {"visualization_data_accessible": False}
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        logger.info("Generating test report...")
        
        # Create report data
        report = {
            "test_id": self.test_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "num_workers": self.num_workers,
                "test_duration": self.test_duration,
                "hardware_profiles": self.hardware_profiles,
                "include_failures": self.include_failures
            },
            "validation_results": self.validation_results,
            "summary": {
                "dashboard_accessible": self.validation_results.get("dashboard", {}).get("dashboard_accessible", False),
                "results_page_accessible": self.validation_results.get("results_page", {}).get("results_page_accessible", False),
                "result_aggregation_working": self.validation_results.get("result_aggregation", {}).get("summary_accessible", False),
                "integration_working": self.validation_results.get("integration", {}).get("visualization_data_accessible", False),
                "overall_success": all([
                    self.validation_results.get("dashboard", {}).get("dashboard_accessible", False),
                    self.validation_results.get("results_page", {}).get("results_page_accessible", False),
                    self.validation_results.get("result_aggregation", {}).get("summary_accessible", False),
                    self.validation_results.get("integration", {}).get("visualization_data_accessible", False)
                ])
            }
        }
        
        # Write report to file
        with open(self.results_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated test report: {self.results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print(f"End-to-End Test Results (ID: {self.test_id})")
        print("="*80)
        print(f"Configuration: {self.num_workers} workers, {self.test_duration}s duration")
        print(f"Hardware Profiles: {', '.join(self.hardware_profiles)}")
        print("-"*80)
        print("Validation Summary:")
        print(f"  Dashboard Accessible: {report['summary']['dashboard_accessible']}")
        print(f"  Results Page Accessible: {report['summary']['results_page_accessible']}")
        print(f"  Result Aggregation Working: {report['summary']['result_aggregation_working']}")
        print(f"  Dashboard-Aggregator Integration: {report['summary']['integration_working']}")
        print("-"*80)
        print(f"Overall Success: {report['summary']['overall_success']}")
        print("="*80)
        print(f"Detailed report saved to: {self.results_file}")
        
        return report
    
    def cleanup(self):
        """Cleanup all processes and resources."""
        logger.info("Cleaning up processes...")
        
        # Terminate all worker processes
        for process in self.worker_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.debug(f"Error terminating worker process: {e}")
        
        # Terminate all service processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.debug(f"Error terminating service process: {e}")
        
        logger.info("Cleanup completed")
    
    async def run(self):
        """Run the end-to-end test."""
        try:
            # Start services
            await self.start_services()
            
            # Start workers
            await self.start_workers()
            
            # Submit test workloads
            await self.submit_test_workloads()
            
            # Inject failures if enabled
            await self.inject_failures()
            
            # Wait for tests to complete
            await self.wait_for_completion()
            
            # Validate dashboard
            await self.validate_dashboard()
            
            # Validate results
            await self.validate_results()
            
            # Validate integration
            await self.validate_integration()
            
            # Generate report
            report = self.generate_report()
            
            return report
        finally:
            # Ensure cleanup happens
            self.cleanup()


async def main():
    """Main entry point for the end-to-end test."""
    parser = argparse.ArgumentParser(description="End-to-End Testing for Distributed Testing Framework")
    parser.add_argument("--workers", type=int, default=5, help="Number of simulated worker nodes")
    parser.add_argument("--test-duration", type=int, default=60, help="Duration of tests in seconds")
    parser.add_argument("--hardware-profiles", type=str, default="all", 
                       help="Hardware profiles to simulate [cpu,gpu,webgpu,webnn,multi] or 'all'")
    parser.add_argument("--include-failures", action="store_true", help="Include simulated failures")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Port for the monitoring dashboard")
    parser.add_argument("--coordinator-port", type=int, default=8081, help="Port for the coordinator service")
    parser.add_argument("--result-aggregator-port", type=int, default=8082, help="Port for the result aggregator service")
    parser.add_argument("--cache-dir", type=str, default="./.e2e_test_cache", help="Directory for caching test data")
    parser.add_argument("--report-dir", type=str, default="./e2e_test_reports", help="Directory for test reports")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without browser interaction")
    
    args = parser.parse_args()
    
    # Parse hardware profiles
    if args.hardware_profiles.lower() == "all":
        hardware_profiles = HARDWARE_TYPES
    else:
        hardware_profiles = args.hardware_profiles.split(",")
    
    # Create and run test framework
    framework = EndToEndTestFramework(
        num_workers=args.workers,
        test_duration=args.test_duration,
        hardware_profiles=hardware_profiles,
        include_failures=args.include_failures,
        dashboard_port=args.dashboard_port,
        coordinator_port=args.coordinator_port,
        result_aggregator_port=args.result_aggregator_port,
        cache_dir=args.cache_dir,
        report_dir=args.report_dir,
        debug=args.debug,
        headless=args.headless
    )
    
    report = await framework.run()
    return report["summary"]["overall_success"]


if __name__ == "__main__":
    # Handle KeyboardInterrupt gracefully
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(130)