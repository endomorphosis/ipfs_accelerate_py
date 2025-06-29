#!/usr/bin/env python3
"""
Circuit Breaker Benchmark Script

This script benchmarks the benefits of using the circuit breaker pattern in the
distributed testing framework. It measures recovery times, success rates, and 
resource utilization with and without the circuit breaker enabled.

The benchmark:
1. Creates a test environment with simulated browser instances
2. Injects a series of controlled failures
3. Measures recovery performance with and without circuit breaker
4. Analyzes and reports the results
5. Generates visualizations comparing the approaches

Usage:
    python benchmark_circuit_breaker.py [--browsers N] [--iterations N] [--report file.json]
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("circuit_breaker_benchmark")

# Import required components
try:
    from selenium_browser_bridge import (
        BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
    )
except ImportError:
    logger.error("Error importing selenium_browser_bridge. Make sure it exists at the expected path.")
    SELENIUM_AVAILABLE = False

try:
    from browser_failure_injector import (
        BrowserFailureInjector, FailureType
    )
    INJECTOR_AVAILABLE = True
except ImportError:
    logger.error("Error importing browser_failure_injector. Make sure it exists at the expected path.")
    INJECTOR_AVAILABLE = False
    
    # Define fallback FailureType for type checking
    from enum import Enum
    class FailureType(Enum):
        """Types of browser failures."""
        CONNECTION_FAILURE = "connection_failure"
        RESOURCE_EXHAUSTION = "resource_exhaustion"
        GPU_ERROR = "gpu_error"
        API_ERROR = "api_error"
        TIMEOUT = "timeout"
        CRASH = "crash"
        INTERNAL_ERROR = "internal_error"
        UNKNOWN = "unknown"

try:
    from circuit_breaker import CircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    logger.error("Error importing circuit_breaker. Make sure it exists at the expected path.")
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from browser_recovery_strategies import BrowserRecoveryManager, RecoveryLevel
    RECOVERY_AVAILABLE = True
except ImportError:
    logger.error("Error importing browser_recovery_strategies. Make sure it exists at the expected path.")
    RECOVERY_AVAILABLE = False
    
    # Define fallback RecoveryLevel for type checking
    from enum import Enum
    class RecoveryLevel(Enum):
        """Recovery levels."""
        MINIMAL = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        CRITICAL = 5

# Try to import visualization dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib or NumPy not available. Visualization will be disabled.")
    VISUALIZATION_AVAILABLE = False

class CircuitBreakerBenchmark:
    """
    Benchmark class for evaluating circuit breaker performance.
    
    This class creates a test environment with simulated browser instances,
    injects controlled failures, and measures recovery performance with
    and without the circuit breaker pattern.
    """
    
    def __init__(self, num_browsers: int = 3, num_iterations: int = 5,
                 failure_types: Optional[List[FailureType]] = None,
                 save_report: Optional[str] = None,
                 simulate: bool = False,
                 verbose: bool = False):
        """
        Initialize the benchmark with the given parameters.
        
        Args:
            num_browsers: Number of browser instances to create
            num_iterations: Number of iterations to run for each test
            failure_types: List of failure types to test (None = all types)
            save_report: Path to save the report (or None)
            simulate: Whether to simulate browser operations
            verbose: Whether to enable verbose logging
        """
        self.num_browsers = num_browsers
        self.num_iterations = num_iterations
        self.save_report = save_report
        self.simulate = simulate or not SELENIUM_AVAILABLE
        self.verbose = verbose
        
        # Set failure types to test
        if failure_types:
            self.failure_types = failure_types
        else:
            # Default to a representative subset of failure types
            self.failure_types = [
                FailureType.CONNECTION_FAILURE,
                FailureType.RESOURCE_EXHAUSTION,
                FailureType.GPU_ERROR,
                FailureType.CRASH
            ]
        
        # Test intensities
        self.intensities = ["mild", "moderate", "severe"]
        
        # Browser types to test
        self.browser_types = ["chrome"]
        if self.num_browsers > 1:
            self.browser_types.append("firefox")
        if self.num_browsers > 2:
            self.browser_types.append("edge")
        
        # Browser platforms to test
        self.platforms = ["webgpu"]
        
        # Results storage
        self.results = {
            "with_circuit_breaker": {},
            "without_circuit_breaker": {},
            "summary": {}
        }
        
        # For resource usage monitoring
        self.start_resources = {}
        self.end_resources = {}
        
        logger.info(f"Initialized circuit breaker benchmark with {num_browsers} browsers, {num_iterations} iterations")
        
        # Set verbose logging if requested
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("circuit_breaker").setLevel(logging.DEBUG)
            logging.getLogger("browser_failure_injector").setLevel(logging.DEBUG)
            logging.getLogger("selenium_browser_bridge").setLevel(logging.DEBUG)
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark and return the results.
        
        Returns:
            Dictionary with benchmark results
        """
        if not CIRCUIT_BREAKER_AVAILABLE or not INJECTOR_AVAILABLE:
            logger.error("Circuit breaker or failure injector not available. Cannot run benchmark.")
            return {"error": "Required components not available"}
        
        logger.info("Starting circuit breaker benchmark")
        
        # Start time for the overall benchmark
        benchmark_start_time = time.time()
        
        # Record system resources at start
        self.start_resources = self._get_resource_usage()
        
        # Run benchmark with circuit breaker enabled
        logger.info("Running benchmark WITH circuit breaker")
        with_cb_results = await self._run_test_suite(use_circuit_breaker=True)
        self.results["with_circuit_breaker"] = with_cb_results
        
        # Short delay between test runs
        await asyncio.sleep(5)
        
        # Run benchmark without circuit breaker
        logger.info("Running benchmark WITHOUT circuit breaker")
        without_cb_results = await self._run_test_suite(use_circuit_breaker=False)
        self.results["without_circuit_breaker"] = without_cb_results
        
        # Record system resources at end
        self.end_resources = self._get_resource_usage()
        
        # Calculate summary statistics
        self._calculate_summary()
        
        # Record overall benchmark time
        benchmark_duration = time.time() - benchmark_start_time
        self.results["benchmark_duration_seconds"] = benchmark_duration
        
        logger.info(f"Benchmark completed in {benchmark_duration:.1f} seconds")
        
        # Generate visualizations
        if VISUALIZATION_AVAILABLE:
            self._generate_visualizations()
        
        # Save report if requested
        if self.save_report:
            self._save_report()
        
        return self.results
    
    async def _run_test_suite(self, use_circuit_breaker: bool) -> Dict[str, Any]:
        """
        Run a complete test suite with or without circuit breaker.
        
        Args:
            use_circuit_breaker: Whether to use circuit breaker for the test
            
        Returns:
            Dictionary with test results
        """
        results = {
            "browser_results": {},
            "failure_type_results": {},
            "intensity_results": {},
            "overall": {
                "recovery_times_ms": [],
                "success_rate": 0,
                "resource_usage": {}
            }
        }
        
        # Get starting resources for this test
        start_resources = self._get_resource_usage()
        
        # Create test cases: combinations of browser_type, failure_type, and intensity
        test_cases = []
        for browser_type in self.browser_types:
            for platform in self.platforms:
                for failure_type in self.failure_types:
                    for intensity in self.intensities:
                        # Skip severe intensity for the first half of iterations when not using circuit breaker
                        # This prevents excessive cascading failures that could impact benchmark accuracy
                        if not use_circuit_breaker and intensity == "severe" and len(test_cases) < len(self.browser_types) * len(self.platforms) * len(self.failure_types) * len(self.intensities) / 2:
                            continue
                            
                        test_cases.append({
                            "browser_type": browser_type,
                            "platform": platform,
                            "failure_type": failure_type,
                            "intensity": intensity
                        })
        
        # Run iterations of each test case
        for i in range(self.num_iterations):
            logger.info(f"Running iteration {i+1}/{self.num_iterations} {'WITH' if use_circuit_breaker else 'WITHOUT'} circuit breaker")
            
            # Shuffle test cases for more realistic testing
            import random
            random.shuffle(test_cases)
            
            for test_case in test_cases:
                browser_type = test_case["browser_type"]
                platform = test_case["platform"]
                failure_type = test_case["failure_type"]
                intensity = test_case["intensity"]
                
                # Run single test case
                logger.info(f"Testing {browser_type} + {platform} with {failure_type.value} ({intensity}) {'WITH' if use_circuit_breaker else 'WITHOUT'} circuit breaker")
                
                test_result = await self._run_test_case(
                    browser_type=browser_type,
                    platform=platform,
                    failure_type=failure_type,
                    intensity=intensity,
                    use_circuit_breaker=use_circuit_breaker
                )
                
                # Add results to appropriate categories
                self._add_to_category(results, "browser_results", f"{browser_type}_{platform}", test_result)
                self._add_to_category(results, "failure_type_results", failure_type.value, test_result)
                self._add_to_category(results, "intensity_results", intensity, test_result)
                
                # Add to overall results
                if test_result["recovery_success"]:
                    results["overall"]["recovery_times_ms"].append(test_result["recovery_time_ms"])
                
                # Short delay between tests to allow system to stabilize
                await asyncio.sleep(1)
        
        # Calculate overall success rate
        total_tests = len(test_cases) * self.num_iterations
        successful_recoveries = sum(1 for time in results["overall"]["recovery_times_ms"])
        results["overall"]["success_rate"] = successful_recoveries / total_tests
        
        # Calculate average recovery time
        if results["overall"]["recovery_times_ms"]:
            results["overall"]["avg_recovery_time_ms"] = statistics.mean(results["overall"]["recovery_times_ms"])
            results["overall"]["median_recovery_time_ms"] = statistics.median(results["overall"]["recovery_times_ms"])
            results["overall"]["min_recovery_time_ms"] = min(results["overall"]["recovery_times_ms"])
            results["overall"]["max_recovery_time_ms"] = max(results["overall"]["recovery_times_ms"])
            
            if len(results["overall"]["recovery_times_ms"]) > 1:
                results["overall"]["stdev_recovery_time_ms"] = statistics.stdev(results["overall"]["recovery_times_ms"])
        
        # Calculate category averages
        for category_name in ["browser_results", "failure_type_results", "intensity_results"]:
            for key in results[category_name]:
                if results[category_name][key]["recovery_times_ms"]:
                    results[category_name][key]["avg_recovery_time_ms"] = statistics.mean(results[category_name][key]["recovery_times_ms"])
                    results[category_name][key]["success_rate"] = results[category_name][key]["successes"] / results[category_name][key]["total"]
        
        # Get ending resources for this test
        end_resources = self._get_resource_usage()
        
        # Calculate resource usage
        results["overall"]["resource_usage"] = self._calculate_resource_diff(start_resources, end_resources)
        
        return results
    
    async def _run_test_case(self, browser_type: str, platform: str, 
                            failure_type: FailureType, intensity: str,
                            use_circuit_breaker: bool) -> Dict[str, Any]:
        """
        Run a single test case with the specified parameters.
        
        Args:
            browser_type: Type of browser to use (chrome, firefox, edge)
            platform: Platform to use (webgpu, webnn)
            failure_type: Type of failure to inject
            intensity: Intensity of failure (mild, moderate, severe)
            use_circuit_breaker: Whether to use circuit breaker for recovery
            
        Returns:
            Dictionary with test results
        """
        result = {
            "browser_type": browser_type,
            "platform": platform,
            "failure_type": failure_type.value,
            "intensity": intensity,
            "use_circuit_breaker": use_circuit_breaker,
            "recovery_success": False,
            "recovery_time_ms": 0,
            "error": None
        }
        
        # Create browser configuration
        config = BrowserConfiguration(
            browser_name=browser_type,
            platform=platform,
            headless=True,
            timeout=30
        )
        
        # Create browser bridge
        bridge = SeleniumBrowserBridge(config)
        
        # Create circuit breaker if enabled
        circuit_breaker = None
        if use_circuit_breaker:
            circuit_breaker = CircuitBreaker(
                failure_threshold=3,     # Open after 3 failures
                recovery_timeout=10,     # Stay open for 10 seconds
                half_open_after=5,       # Try half-open after 5 seconds
                name="benchmark_circuit"
            )
        
        try:
            # Launch browser with simulation if needed
            launch_success = await bridge.launch(allow_simulation=self.simulate)
            
            if not launch_success:
                logger.error(f"Failed to launch {browser_type}")
                result["error"] = "Failed to launch browser"
                return result
            
            # Create recovery manager
            recovery_manager = BrowserRecoveryManager(
                circuit_breaker=circuit_breaker if use_circuit_breaker else None
            )
            
            # Create failure injector with circuit breaker if enabled
            injector = BrowserFailureInjector(
                bridge, 
                circuit_breaker=circuit_breaker if use_circuit_breaker else None,
                use_circuit_breaker=use_circuit_breaker
            )
            
            # Inject failure
            logger.info(f"Injecting {failure_type.value} failure with {intensity} intensity")
            injection_result = await injector.inject_failure(failure_type, intensity)
            
            if not injection_result.get("success", False):
                logger.warning(f"Failed to inject {failure_type.value} failure")
                result["error"] = f"Failed to inject failure: {injection_result.get('error', 'unknown error')}"
                await bridge.close()
                return result
            
            # Attempt recovery
            start_time = time.time()
            
            # Determine recovery level based on intensity and circuit breaker
            if intensity == "severe":
                recovery_level = RecoveryLevel.HIGH
            elif intensity == "moderate":
                recovery_level = RecoveryLevel.MEDIUM
            else:
                recovery_level = RecoveryLevel.LOW
                
            # If using circuit breaker, adjust level based on circuit state
            if use_circuit_breaker and circuit_breaker:
                circuit_state = circuit_breaker.get_state()
                if circuit_state == "open":
                    recovery_level = RecoveryLevel.HIGH
                elif circuit_state == "half-open":
                    recovery_level = RecoveryLevel.MEDIUM
            
            # Execute recovery
            recovery_success = await recovery_manager.recover_browser(
                bridge=bridge,
                failure_type=failure_type,
                recovery_level=recovery_level,
                max_attempts=3,
                retry_delay=1
            )
            
            # Record recovery time
            recovery_time_ms = (time.time() - start_time) * 1000
            
            # Update result
            result["recovery_success"] = recovery_success
            result["recovery_time_ms"] = recovery_time_ms
            
            # Check if we can still perform a basic operation
            if recovery_success:
                try:
                    driver = getattr(bridge, "driver", None)
                    if driver and not self.simulate:
                        driver.get("about:blank")
                        title = driver.title
                        logger.info(f"Browser functional after recovery, title: {title}")
                    else:
                        logger.info("Browser recovery validated (simulation mode)")
                except Exception as e:
                    logger.warning(f"Browser still has issues after recovery: {str(e)}")
                    result["recovery_success"] = False
                    result["error"] = f"Browser still has issues after recovery: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error during test: {str(e)}")
            result["error"] = str(e)
        
        finally:
            # Close browser
            try:
                await bridge.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {str(e)}")
        
        return result
    
    def _add_to_category(self, results: Dict[str, Any], category: str, key: str, test_result: Dict[str, Any]) -> None:
        """
        Add test result to the specified category, initializing if needed.
        
        Args:
            results: Results dictionary to update
            category: Category name (e.g., "browser_results")
            key: Key within the category (e.g., "chrome_webgpu")
            test_result: Test result to add
        """
        if key not in results[category]:
            results[category][key] = {
                "recovery_times_ms": [],
                "successes": 0,
                "total": 0
            }
            
        if test_result["recovery_success"]:
            results[category][key]["recovery_times_ms"].append(test_result["recovery_time_ms"])
            results[category][key]["successes"] += 1
            
        results[category][key]["total"] += 1
    
    def _calculate_summary(self) -> None:
        """Calculate summary statistics comparing with/without circuit breaker."""
        with_cb = self.results["with_circuit_breaker"]["overall"]
        without_cb = self.results["without_circuit_breaker"]["overall"]
        
        summary = {}
        
        # Only calculate if both have data
        if with_cb.get("recovery_times_ms") and without_cb.get("recovery_times_ms"):
            # Recovery time improvement
            with_time = with_cb.get("avg_recovery_time_ms", 0)
            without_time = without_cb.get("avg_recovery_time_ms", 0)
            
            if without_time > 0:
                time_improvement_pct = ((without_time - with_time) / without_time) * 100
                summary["recovery_time_improvement_pct"] = time_improvement_pct
                
            # Success rate improvement
            with_rate = with_cb.get("success_rate", 0)
            without_rate = without_cb.get("success_rate", 0)
            
            if without_rate > 0:
                rate_improvement_pct = ((with_rate - without_rate) / without_rate) * 100
                summary["success_rate_improvement_pct"] = rate_improvement_pct
            else:
                rate_improvement_pct = 100  # Avoid division by zero
                summary["success_rate_improvement_pct"] = rate_improvement_pct
            
            # Resource usage difference
            with_resources = with_cb.get("resource_usage", {})
            without_resources = without_cb.get("resource_usage", {})
            
            if with_resources and without_resources:
                resource_diff = {}
                
                for key in with_resources:
                    if key in without_resources:
                        diff = with_resources[key] - without_resources[key]
                        pct_diff = (diff / without_resources[key]) * 100 if without_resources[key] != 0 else 0
                        resource_diff[key] = {
                            "difference": diff,
                            "percent_diff": pct_diff
                        }
                
                summary["resource_usage_diff"] = resource_diff
        
        # Category comparisons
        summary["categories"] = {}
        
        for category in ["browser_results", "failure_type_results", "intensity_results"]:
            category_summary = {}
            
            for key in self.results["with_circuit_breaker"][category]:
                if key in self.results["without_circuit_breaker"][category]:
                    with_data = self.results["with_circuit_breaker"][category][key]
                    without_data = self.results["without_circuit_breaker"][category][key]
                    
                    # Skip if no recovery times
                    if not with_data.get("recovery_times_ms") or not without_data.get("recovery_times_ms"):
                        continue
                    
                    # Calculate improvements
                    with_time = with_data.get("avg_recovery_time_ms", 0)
                    without_time = without_data.get("avg_recovery_time_ms", 0)
                    
                    with_rate = with_data.get("success_rate", 0)
                    without_rate = without_data.get("success_rate", 0)
                    
                    if without_time > 0:
                        time_improvement_pct = ((without_time - with_time) / without_time) * 100
                    else:
                        time_improvement_pct = 0
                        
                    if without_rate > 0:
                        rate_improvement_pct = ((with_rate - without_rate) / without_rate) * 100
                    else:
                        rate_improvement_pct = 100  # Avoid division by zero
                    
                    category_summary[key] = {
                        "recovery_time_improvement_pct": time_improvement_pct,
                        "success_rate_improvement_pct": rate_improvement_pct
                    }
            
            summary["categories"][category] = category_summary
        
        # Overall rating
        overall_rating = "Undetermined"
        if "recovery_time_improvement_pct" in summary and "success_rate_improvement_pct" in summary:
            time_improvement = summary["recovery_time_improvement_pct"]
            rate_improvement = summary["success_rate_improvement_pct"]
            
            if time_improvement > 30 and rate_improvement > 30:
                overall_rating = "Excellent"
            elif time_improvement > 20 and rate_improvement > 20:
                overall_rating = "Very Good"
            elif time_improvement > 10 and rate_improvement > 10:
                overall_rating = "Good"
            elif time_improvement > 0 and rate_improvement > 0:
                overall_rating = "Positive"
            elif time_improvement < 0 and rate_improvement < 0:
                overall_rating = "Negative"
            else:
                overall_rating = "Mixed"
                
        summary["overall_rating"] = overall_rating
        
        # Store in results
        self.results["summary"] = summary
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with resource metrics
        """
        resources = {}
        
        try:
            import psutil
            
            # CPU usage
            resources["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            resources["memory_percent"] = memory.percent
            resources["memory_used_mb"] = memory.used / (1024 * 1024)
            
            # Process-specific metrics
            process = psutil.Process()
            resources["process_cpu_percent"] = process.cpu_percent(interval=0.1)
            resources["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            resources["process_threads"] = process.num_threads()
            
            # Other metrics
            resources["system_uptime"] = time.time() - psutil.boot_time()
            resources["process_open_files"] = len(process.open_files())
            
        except (ImportError, Exception) as e:
            logger.warning(f"Error getting resource usage: {str(e)}")
            # Provide default metrics if psutil not available
            resources["cpu_percent"] = 0
            resources["memory_percent"] = 0
            resources["memory_used_mb"] = 0
            resources["process_cpu_percent"] = 0
            resources["process_memory_mb"] = 0
            resources["process_threads"] = 0
        
        return resources
    
    def _calculate_resource_diff(self, start: Dict[str, float], end: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate difference in resource usage.
        
        Args:
            start: Starting resource measurements
            end: Ending resource measurements
            
        Returns:
            Dictionary with resource differences
        """
        diff = {}
        
        for key in end:
            if key in start:
                diff[key] = end[key] - start[key]
        
        return diff
    
    def _generate_visualizations(self) -> None:
        """Generate visualizations of benchmark results."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization dependencies not available. Cannot generate visualizations.")
            return
            
        try:
            # Create directory for visualizations
            os.makedirs("benchmark_reports", exist_ok=True)
            
            # Generate recovery time comparison graph
            self._generate_recovery_time_graph()
            
            # Generate success rate comparison graph
            self._generate_success_rate_graph()
            
            # Generate category comparisons
            self._generate_category_graphs()
            
            # Generate summary dashboard
            self._generate_summary_dashboard()
            
            logger.info(f"Visualizations saved to benchmark_reports/ directory")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def _generate_recovery_time_graph(self) -> None:
        """Generate a graph comparing recovery times with and without circuit breaker."""
        plt.figure(figsize=(10, 6))
        
        with_cb = self.results["with_circuit_breaker"]["overall"].get("recovery_times_ms", [])
        without_cb = self.results["without_circuit_breaker"]["overall"].get("recovery_times_ms", [])
        
        if not with_cb or not without_cb:
            logger.warning("Insufficient data for recovery time comparison")
            return
        
        # Create box plot
        plt.boxplot(
            [with_cb, without_cb],
            labels=["With Circuit Breaker", "Without Circuit Breaker"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="navy")
        )
        
        # Add mean values
        with_mean = statistics.mean(with_cb) if with_cb else 0
        without_mean = statistics.mean(without_cb) if without_cb else 0
        
        plt.scatter([1], [with_mean], color='red', marker='X', s=100, label='Mean')
        plt.scatter([2], [without_mean], color='red', marker='X', s=100)
        
        # Calculate improvement
        if without_mean > 0:
            improvement = ((without_mean - with_mean) / without_mean) * 100
            title = f"Recovery Time Comparison\n{improvement:.1f}% improvement with Circuit Breaker"
        else:
            title = "Recovery Time Comparison"
            
        plt.title(title)
        plt.ylabel("Recovery Time (ms)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"benchmark_reports/recovery_time_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_success_rate_graph(self) -> None:
        """Generate a graph comparing success rates with and without circuit breaker."""
        plt.figure(figsize=(10, 6))
        
        with_rate = self.results["with_circuit_breaker"]["overall"].get("success_rate", 0) * 100
        without_rate = self.results["without_circuit_breaker"]["overall"].get("success_rate", 0) * 100
        
        plt.bar([0, 1], [with_rate, without_rate], color=['green', 'orange'])
        
        plt.title("Recovery Success Rate Comparison")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 105)  # Leave room for text labels
        plt.xticks([0, 1], ["With Circuit Breaker", "Without Circuit Breaker"])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage labels
        plt.text(0, with_rate + 2, f"{with_rate:.1f}%", ha='center')
        plt.text(1, without_rate + 2, f"{without_rate:.1f}%", ha='center')
        
        # Add improvement calculation
        if without_rate > 0:
            improvement = ((with_rate - without_rate) / without_rate) * 100
            plt.figtext(0.5, 0.02, f"{improvement:.1f}% improvement with Circuit Breaker", ha='center', fontsize=12)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"benchmark_reports/success_rate_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_category_graphs(self) -> None:
        """Generate graphs comparing performance across different categories."""
        # Generate intensity comparison
        self._generate_category_graph(
            category="intensity_results",
            title="Recovery Time by Failure Intensity",
            ylabel="Average Recovery Time (ms)"
        )
        
        # Generate failure type comparison
        self._generate_category_graph(
            category="failure_type_results",
            title="Recovery Time by Failure Type",
            ylabel="Average Recovery Time (ms)"
        )
        
        # Generate browser comparison
        self._generate_category_graph(
            category="browser_results",
            title="Recovery Time by Browser",
            ylabel="Average Recovery Time (ms)"
        )
    
    def _generate_category_graph(self, category: str, title: str, ylabel: str) -> None:
        """
        Generate a graph comparing performance across a specific category.
        
        Args:
            category: Category to compare (e.g., "browser_results")
            title: Graph title
            ylabel: Y-axis label
        """
        plt.figure(figsize=(12, 6))
        
        with_data = self.results["with_circuit_breaker"][category]
        without_data = self.results["without_circuit_breaker"][category]
        
        # Get common categories for comparison
        common_keys = [key for key in with_data if key in without_data]
        
        if not common_keys:
            logger.warning(f"No common {category} for comparison")
            return
        
        # Prepare data
        with_times = []
        without_times = []
        labels = []
        
        for key in common_keys:
            with_avg = with_data[key].get("avg_recovery_time_ms", 0)
            without_avg = without_data[key].get("avg_recovery_time_ms", 0)
            
            with_times.append(with_avg)
            without_times.append(without_avg)
            labels.append(key)
        
        # Set up plot
        x = np.arange(len(labels))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, with_times, width, label='With Circuit Breaker', color='green')
        plt.bar(x + width/2, without_times, width, label='Without Circuit Breaker', color='orange')
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"benchmark_reports/{category}_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_dashboard(self) -> None:
        """Generate a summary dashboard with key metrics."""
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        
        # Recovery Time Comparison
        with_cb_time = self.results["with_circuit_breaker"]["overall"].get("avg_recovery_time_ms", 0)
        without_cb_time = self.results["without_circuit_breaker"]["overall"].get("avg_recovery_time_ms", 0)
        
        plt.bar([0, 1], [with_cb_time, without_cb_time], color=['green', 'orange'])
        plt.title("Average Recovery Time (ms)")
        plt.xticks([0, 1], ["With CB", "Without CB"])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Success Rate Comparison
        plt.subplot(2, 2, 2)
        
        with_cb_rate = self.results["with_circuit_breaker"]["overall"].get("success_rate", 0) * 100
        without_cb_rate = self.results["without_circuit_breaker"]["overall"].get("success_rate", 0) * 100
        
        plt.bar([0, 1], [with_cb_rate, without_cb_rate], color=['green', 'orange'])
        plt.title("Success Rate (%)")
        plt.xticks([0, 1], ["With CB", "Without CB"])
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Intensity Comparison
        plt.subplot(2, 2, 3)
        
        intensities = ['mild', 'moderate', 'severe']
        with_intensity_times = []
        without_intensity_times = []
        
        for intensity in intensities:
            with_data = self.results["with_circuit_breaker"]["intensity_results"].get(intensity, {})
            without_data = self.results["without_circuit_breaker"]["intensity_results"].get(intensity, {})
            
            with_intensity_times.append(with_data.get("avg_recovery_time_ms", 0))
            without_intensity_times.append(without_data.get("avg_recovery_time_ms", 0))
        
        x = np.arange(len(intensities))
        width = 0.35
        
        plt.bar(x - width/2, with_intensity_times, width, label='With CB', color='green')
        plt.bar(x + width/2, without_intensity_times, width, label='Without CB', color='orange')
        
        plt.title("Recovery Time by Intensity")
        plt.xticks(x, intensities)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Improvement Summary
        plt.subplot(2, 2, 4)
        
        summary = self.results.get("summary", {})
        
        time_improvement = summary.get("recovery_time_improvement_pct", 0)
        rate_improvement = summary.get("success_rate_improvement_pct", 0)
        
        plt.bar([0, 1], [time_improvement, rate_improvement], color=['blue', 'purple'])
        plt.title("Improvement with Circuit Breaker (%)")
        plt.xticks([0, 1], ["Recovery Time", "Success Rate"])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Overall title
        plt.suptitle("Circuit Breaker Performance Summary", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Add overall rating at the bottom
        overall_rating = summary.get("overall_rating", "Undetermined")
        plt.figtext(0.5, 0.01, f"Overall Rating: {overall_rating}", ha='center', fontsize=14)
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"benchmark_reports/circuit_breaker_summary_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_report(self) -> None:
        """Save benchmark results to a JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.save_report)), exist_ok=True)
            
            # Add timestamp and environment info to results
            self.results["timestamp"] = datetime.now().isoformat()
            self.results["environment"] = {
                "python_version": sys.version,
                "os": sys.platform,
                "simulate": self.simulate
            }
            
            # Write to file
            with open(self.save_report, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Results saved to {self.save_report}")
            
            # Also generate markdown report
            self._generate_markdown_report()
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
    
    def _generate_markdown_report(self) -> None:
        """Generate a markdown report of benchmark results."""
        try:
            # Create markdown file path from JSON path
            markdown_path = self.save_report.replace('.json', '.md')
            if markdown_path == self.save_report:
                markdown_path += '.md'
                
            with open(markdown_path, 'w') as f:
                # Title and overview
                f.write("# Circuit Breaker Benchmark Report\n\n")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Generated: {timestamp}\n\n")
                
                # Summary section
                f.write("## Executive Summary\n\n")
                
                summary = self.results.get("summary", {})
                overall_rating = summary.get("overall_rating", "Undetermined")
                
                f.write(f"**Overall Rating:** {overall_rating}\n\n")
                
                if "recovery_time_improvement_pct" in summary:
                    time_improvement = summary["recovery_time_improvement_pct"]
                    f.write(f"**Recovery Time Improvement:** {time_improvement:.1f}%\n\n")
                    
                if "success_rate_improvement_pct" in summary:
                    rate_improvement = summary["success_rate_improvement_pct"]
                    f.write(f"**Success Rate Improvement:** {rate_improvement:.1f}%\n\n")
                
                # Configuration
                f.write("## Benchmark Configuration\n\n")
                f.write(f"- Browsers tested: {', '.join(self.browser_types)}\n")
                f.write(f"- Platforms tested: {', '.join(self.platforms)}\n")
                f.write(f"- Failure types tested: {', '.join([f.value for f in self.failure_types])}\n")
                f.write(f"- Intensities tested: {', '.join(self.intensities)}\n")
                f.write(f"- Iterations: {self.num_iterations}\n")
                f.write(f"- Simulation mode: {'Yes' if self.simulate else 'No'}\n\n")
                
                # Results tables
                f.write("## Performance Comparison\n\n")
                
                # Recovery time table
                f.write("### Recovery Time (ms)\n\n")
                f.write("| Metric | With Circuit Breaker | Without Circuit Breaker | Improvement |\n")
                f.write("|--------|---------------------|------------------------|-------------|\n")
                
                with_cb = self.results["with_circuit_breaker"]["overall"]
                without_cb = self.results["without_circuit_breaker"]["overall"]
                
                # Average recovery time
                with_avg = with_cb.get("avg_recovery_time_ms", 0)
                without_avg = without_cb.get("avg_recovery_time_ms", 0)
                
                if without_avg > 0:
                    improvement = ((without_avg - with_avg) / without_avg) * 100
                    improvement_str = f"{improvement:.1f}%"
                else:
                    improvement_str = "N/A"
                    
                f.write(f"| Average | {with_avg:.1f} | {without_avg:.1f} | {improvement_str} |\n")
                
                # Median recovery time
                with_median = with_cb.get("median_recovery_time_ms", 0)
                without_median = without_cb.get("median_recovery_time_ms", 0)
                
                if without_median > 0:
                    improvement = ((without_median - with_median) / without_median) * 100
                    improvement_str = f"{improvement:.1f}%"
                else:
                    improvement_str = "N/A"
                    
                f.write(f"| Median | {with_median:.1f} | {without_median:.1f} | {improvement_str} |\n")
                
                # Min recovery time
                with_min = with_cb.get("min_recovery_time_ms", 0)
                without_min = without_cb.get("min_recovery_time_ms", 0)
                
                if without_min > 0:
                    improvement = ((without_min - with_min) / without_min) * 100
                    improvement_str = f"{improvement:.1f}%"
                else:
                    improvement_str = "N/A"
                    
                f.write(f"| Minimum | {with_min:.1f} | {without_min:.1f} | {improvement_str} |\n")
                
                # Max recovery time
                with_max = with_cb.get("max_recovery_time_ms", 0)
                without_max = without_cb.get("max_recovery_time_ms", 0)
                
                if without_max > 0:
                    improvement = ((without_max - with_max) / without_max) * 100
                    improvement_str = f"{improvement:.1f}%"
                else:
                    improvement_str = "N/A"
                    
                f.write(f"| Maximum | {with_max:.1f} | {without_max:.1f} | {improvement_str} |\n\n")
                
                # Success rate table
                f.write("### Success Rate (%)\n\n")
                
                with_rate = with_cb.get("success_rate", 0) * 100
                without_rate = without_cb.get("success_rate", 0) * 100
                
                if without_rate > 0:
                    improvement = ((with_rate - without_rate) / without_rate) * 100
                    improvement_str = f"{improvement:.1f}%"
                else:
                    improvement_str = "N/A"
                
                f.write("| With Circuit Breaker | Without Circuit Breaker | Improvement |\n")
                f.write("|---------------------|------------------------|-------------|\n")
                f.write(f"| {with_rate:.1f}% | {without_rate:.1f}% | {improvement_str} |\n\n")
                
                # Category breakdowns
                self._write_category_breakdown(f, "Failure Type", "failure_type_results")
                self._write_category_breakdown(f, "Intensity", "intensity_results")
                self._write_category_breakdown(f, "Browser", "browser_results")
                
                # Resource usage
                f.write("## Resource Usage\n\n")
                
                with_resources = with_cb.get("resource_usage", {})
                without_resources = without_cb.get("resource_usage", {})
                
                f.write("| Metric | With Circuit Breaker | Without Circuit Breaker | Difference |\n")
                f.write("|--------|---------------------|------------------------|------------|\n")
                
                for key in with_resources:
                    if key in without_resources:
                        with_value = with_resources[key]
                        without_value = without_resources[key]
                        diff = with_value - without_value
                        
                        f.write(f"| {key} | {with_value:.2f} | {without_value:.2f} | {diff:.2f} |\n")
                
                f.write("\n## Conclusion\n\n")
                
                # Generate conclusion based on results
                if overall_rating in ["Excellent", "Very Good"]:
                    f.write("The circuit breaker pattern demonstrates significant benefits for fault tolerance and recovery in the distributed testing framework. It substantially improves recovery times and success rates while providing system stability during failure conditions.\n\n")
                elif overall_rating in ["Good", "Positive"]:
                    f.write("The circuit breaker pattern shows measurable benefits for fault tolerance and recovery in the distributed testing framework. It improves recovery times and success rates while helping maintain system stability during failures.\n\n")
                elif overall_rating == "Mixed":
                    f.write("The circuit breaker pattern shows mixed results in this benchmark. While there are some benefits in certain scenarios, the overall impact is not consistently positive across all test cases.\n\n")
                else:
                    f.write("The benchmark results are inconclusive regarding the benefits of the circuit breaker pattern in this specific implementation and test environment.\n\n")
                
                # Add visualization references if available
                if VISUALIZATION_AVAILABLE:
                    f.write("## Visualizations\n\n")
                    f.write("Detailed visualizations have been generated in the `benchmark_reports/` directory, including:\n\n")
                    f.write("- Recovery time comparison\n")
                    f.write("- Success rate comparison\n")
                    f.write("- Category-specific performance breakdowns\n")
                    f.write("- Summary dashboard\n\n")
                
                logger.info(f"Markdown report generated: {markdown_path}")
                
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
    
    def _write_category_breakdown(self, file, category_name: str, category_key: str) -> None:
        """
        Write a category breakdown to the markdown report.
        
        Args:
            file: File object to write to
            category_name: Human-readable category name
            category_key: Key in results dict for category
        """
        file.write(f"### {category_name} Breakdown\n\n")
        
        with_data = self.results["with_circuit_breaker"][category_key]
        without_data = self.results["without_circuit_breaker"][category_key]
        
        # Get common categories for comparison
        common_keys = [key for key in with_data if key in without_data]
        
        if not common_keys:
            file.write(f"No common {category_name.lower()} data available for comparison.\n\n")
            return
        
        # Setup table
        file.write("| {} | With CB Recovery Time (ms) | Without CB Recovery Time (ms) | Time Improvement | With CB Success Rate | Without CB Success Rate | Rate Improvement |\n".format(category_name))
        file.write("|{}|---------------------|--------------------------|----------------|--------------------|------------------------|----------------|\n".format("-" * len(category_name)))
        
        for key in common_keys:
            with_rec = with_data[key].get("avg_recovery_time_ms", 0)
            without_rec = without_data[key].get("avg_recovery_time_ms", 0)
            
            with_rate = with_data[key].get("success_rate", 0) * 100
            without_rate = without_data[key].get("success_rate", 0) * 100
            
            # Calculate improvements
            if without_rec > 0:
                time_improvement = ((without_rec - with_rec) / without_rec) * 100
                time_improvement_str = f"{time_improvement:.1f}%"
            else:
                time_improvement_str = "N/A"
                
            if without_rate > 0:
                rate_improvement = ((with_rate - without_rate) / without_rate) * 100
                rate_improvement_str = f"{rate_improvement:.1f}%"
            else:
                rate_improvement_str = "N/A"
            
            file.write(f"| {key} | {with_rec:.1f} | {without_rec:.1f} | {time_improvement_str} | {with_rate:.1f}% | {without_rate:.1f}% | {rate_improvement_str} |\n")
        
        file.write("\n")

async def main():
    """
    Main function to parse arguments and run the benchmark.
    """
    parser = argparse.ArgumentParser(description="Benchmark circuit breaker benefits")
    parser.add_argument("--browsers", type=int, default=2, 
                       help="Number of browser types to include in test")
    parser.add_argument("--iterations", type=int, default=3, 
                       help="Number of iterations for each test case")
    parser.add_argument("--failure-types", nargs="+", default=None,
                       help="Specific failure types to test (default: a representative subset)")
    parser.add_argument("--report", type=str, default="benchmark_reports/circuit_breaker_benchmark.json", 
                       help="Path to save JSON report")
    parser.add_argument("--simulate", action="store_true", 
                       help="Run in simulation mode (no real browsers)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up failure types if provided
    failure_types = None
    if args.failure_types:
        try:
            failure_types = [getattr(FailureType, ft.upper()) for ft in args.failure_types]
        except (AttributeError, Exception) as e:
            logger.error(f"Invalid failure type: {e}")
            valid_types = [ft.name for ft in FailureType]
            logger.error(f"Valid failure types: {', '.join(valid_types)}")
            return 1
    
    # Check dependencies
    if not CIRCUIT_BREAKER_AVAILABLE:
        logger.error("Circuit breaker not available. Cannot run benchmark.")
        return 1
        
    if not INJECTOR_AVAILABLE:
        logger.error("Browser failure injector not available. Cannot run benchmark.")
        return 1
    
    if not SELENIUM_AVAILABLE and not args.simulate:
        logger.warning("Selenium not available. Using simulation mode.")
        args.simulate = True
    
    # Create benchmark
    benchmark = CircuitBreakerBenchmark(
        num_browsers=args.browsers,
        num_iterations=args.iterations,
        failure_types=failure_types,
        save_report=args.report,
        simulate=args.simulate,
        verbose=args.verbose
    )
    
    # Run benchmark
    results = await benchmark.run_benchmark()
    
    # Print summary
    time_improvement = results.get("summary", {}).get("recovery_time_improvement_pct", 0)
    rate_improvement = results.get("summary", {}).get("success_rate_improvement_pct", 0)
    overall_rating = results.get("summary", {}).get("overall_rating", "Undetermined")
    
    print("\n" + "=" * 80)
    print("Circuit Breaker Benchmark Results")
    print("=" * 80)
    print(f"Recovery time improvement: {time_improvement:.1f}%")
    print(f"Success rate improvement: {rate_improvement:.1f}%")
    print(f"Overall rating: {overall_rating}")
    print("=" * 80)
    
    # Print report location
    if args.report:
        print(f"\nDetailed report saved to: {args.report}")
        print(f"Markdown report saved to: {args.report.replace('.json', '.md')}")
        
    if VISUALIZATION_AVAILABLE:
        print(f"Visualizations saved to: benchmark_reports/ directory")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)