#!/usr/bin/env python
"""
Advanced API Distributed Testing Framework Runner

This script provides a comprehensive system for running advanced distributed API tests with
features like:

1. Multi-API testing across providers (OpenAI, Claude, Groq, etc.)
2. Performance benchmarking with detailed metrics collection
3. Cost tracking and optimization with customizable strategies
4. Advanced visualization of results with statistical analysis
5. Integration with Prometheus/Grafana for monitoring

Usage:
    # Run a comprehensive benchmark across all available APIs
    python run_advanced_api_tests.py benchmark --all-apis
    
    # Compare specific APIs with custom parameters
    python run_advanced_api_tests.py compare --apis openai,claude,groq --test latency --iterations 20
    
    # Run cost-optimized testing
    python run_advanced_api_tests.py optimize --budget 10.0 --test throughput
    
    # Visualize historical performance
    python run_advanced_api_tests.py visualize --metrics latency,throughput --days 30
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import uuid
import random
from enum import Enum

# Add project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API distributed testing components
from test_api_distributed_integration import APIDistributedTesting, APITestType
from api_backend_distributed_scheduler import (
    APIBackendScheduler, APIRateLimitStrategy, APICostStrategy,
    APIBackendProfile, APIRateLimit, APICostProfile, APIPerformanceProfile
)

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_api_tests")


class TestCommand(Enum):
    """Available test commands."""
    BENCHMARK = "benchmark"      # Run benchmarks across APIs
    COMPARE = "compare"          # Compare specific APIs
    OPTIMIZE = "optimize"        # Run cost-optimized tests
    VISUALIZE = "visualize"      # Visualize historical results
    STATUS = "status"            # Show system status
    LIST = "list"                # List available APIs


class AdvancedAPITestingFramework:
    """
    Advanced framework for comprehensive API testing with cost optimization,
    performance analysis, and visualization capabilities.
    """
    
    def __init__(
        self,
        prometheus_port: int = 8005,
        database_path: str = "api_test_results.json",
        config_file: Optional[str] = None,
        resources: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        rate_limit_strategy: APIRateLimitStrategy = APIRateLimitStrategy.ADAPTIVE,
        cost_strategy: APICostStrategy = APICostStrategy.BALANCED
    ):
        """Initialize the advanced API testing framework."""
        self.database_path = database_path
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Initialize core testing framework
        self.api_testing = APIDistributedTesting(
            prometheus_port=prometheus_port,
            resources=self.resources,
            metadata=self.metadata
        )
        
        # Initialize API backend scheduler
        self.scheduler = APIBackendScheduler(
            rate_limit_strategy=rate_limit_strategy,
            cost_strategy=cost_strategy,
            config_file=config_file
        )
        
        # Start components
        self.scheduler.start()
        
        # Load historical results if available
        self.historical_results = self._load_historical_results()
        
        # Initialize test sessions tracking
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = time.time()
        self.session_results = []
        
        logger.info(f"Advanced API Testing Framework initialized with session ID: {self.current_session_id}")
    
    def _load_historical_results(self) -> Dict[str, Any]:
        """Load historical test results from database."""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading historical results: {e}")
        
        # Return empty structure if no data or error
        return {
            "sessions": [],
            "tests": [],
            "metrics": {
                "latency": {},
                "throughput": {},
                "reliability": {},
                "cost": {}
            }
        }
    
    def _save_historical_results(self) -> None:
        """Save historical test results to database."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.database_path)), exist_ok=True)
            
            with open(self.database_path, 'w') as f:
                json.dump(self.historical_results, f, indent=2)
                
            logger.debug(f"Saved historical results to {self.database_path}")
        except Exception as e:
            logger.error(f"Error saving historical results: {e}")
    
    def _record_test_result(self, api_type: str, test_type: str, parameters: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Record a test result to historical database."""
        # Create test record
        test_record = {
            "test_id": str(uuid.uuid4()),
            "session_id": self.current_session_id,
            "timestamp": time.time(),
            "api_type": api_type,
            "test_type": test_type,
            "parameters": parameters,
            "result": result,
            "summary": result.get("summary", {})
        }
        
        # Add to session results
        self.session_results.append(test_record)
        
        # Add to historical results
        self.historical_results["tests"].append(test_record)
        
        # Add to metrics tracking based on test type
        metrics = self.historical_results["metrics"]
        
        if test_type == "latency" and "summary" in result:
            if api_type not in metrics["latency"]:
                metrics["latency"][api_type] = []
            
            metrics["latency"][api_type].append({
                "timestamp": time.time(),
                "avg_latency": result["summary"].get("avg_latency"),
                "min_latency": result["summary"].get("min_latency"),
                "max_latency": result["summary"].get("max_latency"),
                "model": parameters.get("model")
            })
        
        elif test_type == "throughput" and "summary" in result:
            if api_type not in metrics["throughput"]:
                metrics["throughput"][api_type] = []
            
            metrics["throughput"][api_type].append({
                "timestamp": time.time(),
                "requests_per_second": result["summary"].get("requests_per_second"),
                "concurrency": parameters.get("concurrency"),
                "model": parameters.get("model")
            })
        
        elif test_type == "reliability" and "summary" in result:
            if api_type not in metrics["reliability"]:
                metrics["reliability"][api_type] = []
            
            metrics["reliability"][api_type].append({
                "timestamp": time.time(),
                "success_rate": result["summary"].get("success_rate"),
                "model": parameters.get("model")
            })
        
        # Add cost data if available
        if "cost" in result:
            if api_type not in metrics["cost"]:
                metrics["cost"][api_type] = []
            
            metrics["cost"][api_type].append({
                "timestamp": time.time(),
                "cost": result["cost"],
                "model": parameters.get("model"),
                "test_type": test_type
            })
        
        # Save after each result
        self._save_historical_results()
    
    def _finish_session(self) -> Dict[str, Any]:
        """Finish current test session and return summary."""
        session_end_time = time.time()
        session_duration = session_end_time - self.session_start_time
        
        # Create session record
        session_record = {
            "session_id": self.current_session_id,
            "start_time": self.session_start_time,
            "end_time": session_end_time,
            "duration": session_duration,
            "test_count": len(self.session_results),
            "apis_tested": list(set(r["api_type"] for r in self.session_results)),
            "test_types": list(set(r["test_type"] for r in self.session_results))
        }
        
        # Add to historical data
        self.historical_results["sessions"].append(session_record)
        
        # Save historical data
        self._save_historical_results()
        
        return {
            "session_id": self.current_session_id,
            "duration": session_duration,
            "test_count": len(self.session_results),
            "apis_tested": session_record["apis_tested"],
            "test_types": session_record["test_types"]
        }
    
    def stop(self) -> None:
        """Stop the testing framework and clean up."""
        # Finish current session
        session_summary = self._finish_session()
        logger.info(f"Finished test session {session_summary['session_id']} with {session_summary['test_count']} tests")
        
        # Stop components
        self.scheduler.stop()
        self.api_testing.stop()
        
        logger.info("Advanced API Testing Framework stopped")
    
    def run_benchmark(
        self,
        api_types: Optional[List[str]] = None,
        test_types: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        include_all_apis: bool = False,
        include_all_tests: bool = False
    ) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark across specified APIs and test types.
        
        Args:
            api_types: List of API types to benchmark (None for all)
            test_types: List of test types to run (None for all)
            parameters: Test parameters
            include_all_apis: Whether to include all available APIs
            include_all_tests: Whether to include all test types
            
        Returns:
            Benchmark results
        """
        # Set default parameters if not provided
        parameters = parameters or {
            "prompt": "Summarize the main ideas of climate change in 3 paragraphs.",
            "iterations": 5,
            "concurrency": 3,
            "duration": 10
        }
        
        # Get available APIs
        if include_all_apis or not api_types:
            api_info = self.api_testing.get_api_backend_info()
            api_types = api_info["available_api_types"]
        
        # Get available test types
        if include_all_tests or not test_types:
            test_types = [t.value for t in APITestType]
        
        logger.info(f"Starting benchmark with {len(api_types)} APIs and {len(test_types)} test types")
        
        # Initialize benchmark results
        benchmark_results = {
            "timestamp": time.time(),
            "apis_tested": api_types,
            "test_types": test_types,
            "parameters": parameters,
            "results": {},
            "summary": {}
        }
        
        # Run tests for each API and test type
        for api_type in api_types:
            benchmark_results["results"][api_type] = {}
            
            # Register API
            self.api_testing.register_api_backend(api_type)
            
            # Update scheduler with API
            self.scheduler.register_api_backend(api_type)
            
            for test_type in test_types:
                logger.info(f"Running {test_type} test for {api_type}...")
                
                # Customize parameters for test type
                test_parameters = parameters.copy()
                
                try:
                    # Run the test
                    result = self.api_testing.run_distributed_test(
                        api_type=api_type,
                        test_type=test_type,
                        parameters=test_parameters
                    )
                    
                    # Store the result
                    benchmark_results["results"][api_type][test_type] = result
                    
                    # Record for historical tracking
                    self._record_test_result(api_type, test_type, test_parameters, result)
                    
                except Exception as e:
                    logger.error(f"Error running {test_type} test for {api_type}: {e}")
                    benchmark_results["results"][api_type][test_type] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        # Generate summary metrics
        summary = {}
        
        # Latency ranking
        if "latency" in test_types:
            latency_data = {}
            for api, results in benchmark_results["results"].items():
                if "latency" in results and results["latency"].get("status") == "success":
                    avg_latency = results["latency"]["summary"].get("avg_latency")
                    if avg_latency is not None:
                        latency_data[api] = avg_latency
            
            if latency_data:
                # Rank by latency (lowest first)
                latency_ranking = sorted(latency_data.items(), key=lambda x: x[1])
                summary["latency_ranking"] = [api for api, _ in latency_ranking]
                summary["fastest_api"] = latency_ranking[0][0] if latency_ranking else None
        
        # Throughput ranking
        if "throughput" in test_types:
            throughput_data = {}
            for api, results in benchmark_results["results"].items():
                if "throughput" in results and results["throughput"].get("status") == "success":
                    rps = results["throughput"]["summary"].get("requests_per_second")
                    if rps is not None:
                        throughput_data[api] = rps
            
            if throughput_data:
                # Rank by throughput (highest first)
                throughput_ranking = sorted(throughput_data.items(), key=lambda x: x[1], reverse=True)
                summary["throughput_ranking"] = [api for api, _ in throughput_ranking]
                summary["highest_throughput_api"] = throughput_ranking[0][0] if throughput_ranking else None
        
        # Reliability ranking
        if "reliability" in test_types:
            reliability_data = {}
            for api, results in benchmark_results["results"].items():
                if "reliability" in results and results["reliability"].get("status") == "success":
                    success_rate = results["reliability"]["summary"].get("success_rate")
                    if success_rate is not None:
                        reliability_data[api] = success_rate
            
            if reliability_data:
                # Rank by reliability (highest first)
                reliability_ranking = sorted(reliability_data.items(), key=lambda x: x[1], reverse=True)
                summary["reliability_ranking"] = [api for api, _ in reliability_ranking]
                summary["most_reliable_api"] = reliability_ranking[0][0] if reliability_ranking else None
        
        # Add summary to results
        benchmark_results["summary"] = summary
        
        return benchmark_results
    
    def compare_apis(
        self,
        api_types: List[str],
        test_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        visualize: bool = False,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare specific APIs on a particular test type.
        
        Args:
            api_types: List of API types to compare
            test_type: Type of test to run
            parameters: Test parameters
            visualize: Whether to generate visualizations
            output_file: File to save visualizations to
            
        Returns:
            Comparison results
        """
        # Set default parameters if not provided
        parameters = parameters or {
            "prompt": "Explain quantum computing to a 10-year-old child.",
            "iterations": 10 if test_type == "latency" else 5,
            "concurrency": 3,
            "duration": 15
        }
        
        logger.info(f"Comparing APIs: {api_types} on {test_type} test")
        
        # Run the comparison
        comparison_results = self.api_testing.compare_apis(
            api_types=api_types,
            test_type=test_type,
            parameters=parameters
        )
        
        # Record results for each API
        for api_type, result in comparison_results["results_by_api"].items():
            if result.get("status") != "error":
                self._record_test_result(api_type, test_type, parameters, result)
        
        # Generate visualization if requested
        if visualize and VISUALIZATION_AVAILABLE:
            vis_file = self._generate_comparison_visualization(
                comparison_results, test_type, output_file
            )
            comparison_results["visualization_file"] = vis_file
        
        return comparison_results
    
    def run_cost_optimized_tests(
        self,
        budget: float,
        test_types: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        visualize: bool = False,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run tests optimized for a specific budget.
        
        Args:
            budget: Maximum budget for testing in dollars
            test_types: Types of tests to run
            parameters: Test parameters
            visualize: Whether to generate visualizations
            output_file: File to save visualizations to
            
        Returns:
            Test results
        """
        # Set default parameters if not provided
        parameters = parameters or {
            "prompt": "Write a story about a robot discovering emotions.",
            "iterations": 3,
            "concurrency": 2,
            "duration": 10
        }
        
        logger.info(f"Running cost-optimized tests with budget ${budget:.2f}")
        
        # Get available APIs
        api_info = self.api_testing.get_api_backend_info()
        available_apis = api_info["available_api_types"]
        
        # Configure scheduler for cost optimization
        self.scheduler.cost_strategy = APICostStrategy.MINIMIZE_COST
        
        # Initialize results
        optimization_results = {
            "timestamp": time.time(),
            "budget": budget,
            "test_types": test_types,
            "parameters": parameters,
            "apis_tested": [],
            "results": {},
            "costs": {},
            "total_cost": 0.0
        }
        
        # Run tests while tracking costs
        remaining_budget = budget
        
        for test_type in test_types:
            optimization_results["results"][test_type] = {}
            
            for api_type in available_apis:
                # Skip if budget is exhausted
                if remaining_budget <= 0.01:  # 1 cent minimum
                    logger.info(f"Budget exhausted, skipping {api_type} for {test_type}")
                    continue
                
                # Estimate cost for this test
                estimated_cost = self._estimate_test_cost(api_type, test_type, parameters)
                
                # Skip if estimated cost exceeds remaining budget
                if estimated_cost > remaining_budget:
                    logger.info(f"Estimated cost ${estimated_cost:.2f} exceeds remaining budget ${remaining_budget:.2f}")
                    continue
                
                logger.info(f"Running {test_type} test for {api_type} (est. cost: ${estimated_cost:.2f})")
                
                # Register API
                self.api_testing.register_api_backend(api_type)
                
                # Update scheduler with API
                self.scheduler.register_api_backend(api_type)
                
                try:
                    # Run the test
                    result = self.api_testing.run_distributed_test(
                        api_type=api_type,
                        test_type=test_type,
                        parameters=parameters
                    )
                    
                    # Get actual cost from result or use estimate
                    actual_cost = result.get("cost", estimated_cost)
                    
                    # Store the result
                    optimization_results["results"][test_type][api_type] = result
                    optimization_results["costs"][f"{api_type}_{test_type}"] = actual_cost
                    optimization_results["total_cost"] += actual_cost
                    
                    # Add API to tested list if not already there
                    if api_type not in optimization_results["apis_tested"]:
                        optimization_results["apis_tested"].append(api_type)
                    
                    # Update remaining budget
                    remaining_budget -= actual_cost
                    
                    # Record for historical tracking
                    # Add cost information to result
                    result["cost"] = actual_cost
                    self._record_test_result(api_type, test_type, parameters, result)
                    
                except Exception as e:
                    logger.error(f"Error running {test_type} test for {api_type}: {e}")
                    optimization_results["results"][test_type][api_type] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        # Add budget information
        optimization_results["remaining_budget"] = remaining_budget
        optimization_results["budget_utilization"] = (budget - remaining_budget) / budget
        
        # Generate visualization if requested
        if visualize and VISUALIZATION_AVAILABLE:
            vis_file = self._generate_cost_optimization_visualization(
                optimization_results, output_file
            )
            optimization_results["visualization_file"] = vis_file
        
        return optimization_results
    
    def _estimate_test_cost(self, api_type: str, test_type: str, parameters: Dict[str, Any]) -> float:
        """
        Estimate the cost of running a specific test.
        
        Args:
            api_type: Type of API to test
            test_type: Type of test to run
            parameters: Test parameters
            
        Returns:
            Estimated cost in dollars
        """
        # Get cost profile from scheduler
        api_profile = self.scheduler.api_profiles.get(api_type)
        if not api_profile:
            return 0.0  # Unknown cost
        
        cost_profile = api_profile.cost_profile
        
        # Estimate tokens based on prompt
        prompt = parameters.get("prompt", "")
        input_token_estimate = len(prompt.split()) * 1.3  # Rough estimate
        
        # Adjust based on test type
        if test_type == "latency":
            iterations = parameters.get("iterations", 5)
            # Estimate output tokens (typically 2-4x input for most responses)
            output_token_estimate = input_token_estimate * 3
            
            # Calculate cost for all iterations
            total_input_tokens = input_token_estimate * iterations
            total_output_tokens = output_token_estimate * iterations
            
            return cost_profile.estimate_cost(total_input_tokens, total_output_tokens)
            
        elif test_type == "throughput":
            duration = parameters.get("duration", 10)
            concurrency = parameters.get("concurrency", 3)
            
            # Estimate requests per second based on typical latency
            avg_latency = api_profile.performance.get_avg_latency() or 1.0  # Default to 1s if no data
            requests_per_second = min(concurrency / avg_latency, 10)  # Cap at 10 req/s
            
            # Estimate total requests
            total_requests = requests_per_second * duration
            
            # Estimate output tokens (shorter for throughput tests)
            output_token_estimate = input_token_estimate * 2
            
            # Calculate cost
            total_input_tokens = input_token_estimate * total_requests
            total_output_tokens = output_token_estimate * total_requests
            
            return cost_profile.estimate_cost(total_input_tokens, total_output_tokens)
            
        elif test_type == "reliability":
            iterations = parameters.get("iterations", 10)
            
            # Reliability tests use shorter outputs
            output_token_estimate = input_token_estimate * 1.5
            
            # Calculate cost for all iterations
            total_input_tokens = input_token_estimate * iterations
            total_output_tokens = output_token_estimate * iterations
            
            return cost_profile.estimate_cost(total_input_tokens, total_output_tokens)
            
        elif test_type == "concurrency":
            # Similar to throughput but with multiple concurrency levels
            levels = parameters.get("concurrency_levels", [1, 2, 5, 10])
            requests_per_level = parameters.get("requests_per_level", 5)
            
            # Estimate total requests
            total_requests = sum(levels) * requests_per_level
            
            # Estimate output tokens
            output_token_estimate = input_token_estimate * 2
            
            # Calculate cost
            total_input_tokens = input_token_estimate * total_requests
            total_output_tokens = output_token_estimate * total_requests
            
            return cost_profile.estimate_cost(total_input_tokens, total_output_tokens)
        
        # Default estimate for unknown test types
        return 0.5  # $0.50 default estimate
    
    def visualize_historical_performance(
        self,
        metrics: List[str],
        days: int = 30,
        api_types: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize historical performance metrics.
        
        Args:
            metrics: List of metrics to visualize (latency, throughput, reliability, cost)
            days: Number of days to look back
            api_types: Optional list of API types to include
            output_file: File to save visualizations to
            
        Returns:
            Visualization results
        """
        if not VISUALIZATION_AVAILABLE:
            return {
                "status": "error",
                "error": "Visualization libraries not available. Install matplotlib, pandas, numpy, and seaborn."
            }
        
        logger.info(f"Visualizing historical {', '.join(metrics)} metrics for past {days} days")
        
        # Get historical data
        lookback_time = time.time() - (days * 24 * 60 * 60)
        
        # Filter APIs if specified
        if api_types is None:
            # Use all APIs with data
            api_types = set()
            for metric in metrics:
                if metric in self.historical_results["metrics"]:
                    api_types.update(self.historical_results["metrics"][metric].keys())
            api_types = list(api_types)
        
        # Create visualization
        fig = self._create_historical_visualization(metrics, api_types, lookback_time, output_file)
        
        return {
            "status": "success",
            "metrics": metrics,
            "api_types": api_types,
            "days": days,
            "visualization_file": fig
        }
    
    def _create_historical_visualization(
        self,
        metrics: List[str],
        api_types: List[str],
        lookback_time: float,
        output_file: Optional[str] = None
    ) -> str:
        """
        Create historical performance visualization.
        
        Args:
            metrics: List of metrics to visualize
            api_types: List of API types to include
            lookback_time: Timestamp to look back from
            output_file: File to save visualization to
            
        Returns:
            Path to saved visualization
        """
        # Create figure
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics), sharex=True)
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        
        # Create color map for APIs
        api_colors = {}
        colormap = plt.cm.tab10
        for i, api in enumerate(api_types):
            api_colors[api] = colormap(i % 10)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric not in self.historical_results["metrics"]:
                ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{metric.capitalize()} - No Data")
                continue
            
            # Track if any data plotted
            data_plotted = False
            
            for api in api_types:
                if api not in self.historical_results["metrics"][metric]:
                    continue
                
                # Get data
                data = self.historical_results["metrics"][metric][api]
                
                # Filter by lookback time
                data = [d for d in data if d["timestamp"] >= lookback_time]
                
                if not data:
                    continue
                
                # Convert to pandas for easier plotting
                df = pd.DataFrame(data)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                
                # Extract the value to plot
                if metric == "latency":
                    y_values = df["avg_latency"]
                    y_label = "Average Latency (s)"
                    title = "API Latency over Time"
                elif metric == "throughput":
                    y_values = df["requests_per_second"]
                    y_label = "Requests per Second"
                    title = "API Throughput over Time"
                elif metric == "reliability":
                    y_values = df["success_rate"] * 100  # Convert to percentage
                    y_label = "Success Rate (%)"
                    title = "API Reliability over Time"
                elif metric == "cost":
                    y_values = df["cost"]
                    y_label = "Cost ($)"
                    title = "API Cost over Time"
                
                # Plot
                ax.plot(df["datetime"], y_values, 'o-', label=api, color=api_colors[api])
                data_plotted = True
            
            # Configure axis
            if data_plotted:
                ax.set_title(title)
                ax.set_ylabel(y_label)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Format y-axis for percentages
                if metric == "reliability":
                    ax.yaxis.set_major_formatter(PercentFormatter())
            else:
                ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{metric.capitalize()} - No Data")
        
        # Set overall x-axis label
        fig.text(0.5, 0.01, "Date", ha="center")
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.07)
        
        # Save figure
        if output_file:
            output_path = output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"api_performance_{timestamp}.png"
        
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        logger.info(f"Saved historical visualization to {output_path}")
        return output_path
    
    def _generate_comparison_visualization(
        self,
        comparison_results: Dict[str, Any],
        test_type: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate visualization for API comparison.
        
        Args:
            comparison_results: Results from compare_apis
            test_type: Type of test
            output_file: File to save visualization to
            
        Returns:
            Path to saved visualization
        """
        # Extract data for visualization
        api_results = comparison_results["results_by_api"]
        apis = list(api_results.keys())
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot based on test type
        if test_type == "latency":
            # Extract latency values
            avg_latencies = []
            min_latencies = []
            max_latencies = []
            
            for api in apis:
                result = api_results[api]
                if result.get("status") == "success" and "summary" in result:
                    avg_latencies.append(result["summary"].get("avg_latency", 0))
                    min_latencies.append(result["summary"].get("min_latency", 0))
                    max_latencies.append(result["summary"].get("max_latency", 0))
                else:
                    avg_latencies.append(0)
                    min_latencies.append(0)
                    max_latencies.append(0)
            
            # Create bar chart
            x = np.arange(len(apis))
            width = 0.25
            
            axes[0].bar(x - width, min_latencies, width, label='Min Latency')
            axes[0].bar(x, avg_latencies, width, label='Avg Latency')
            axes[0].bar(x + width, max_latencies, width, label='Max Latency')
            
            axes[0].set_title('API Latency Comparison')
            axes[0].set_ylabel('Latency (seconds)')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(apis)
            axes[0].legend()
            
            # Create latency distribution plot
            for i, api in enumerate(apis):
                result = api_results[api]
                if result.get("status") == "success" and "latencies" in result:
                    latencies = [l for l in result["latencies"] if l is not None]
                    if latencies:
                        sns.kdeplot(latencies, ax=axes[1], label=api)
            
            axes[1].set_title('Latency Distribution')
            axes[1].set_xlabel('Latency (seconds)')
            axes[1].set_ylabel('Density')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        elif test_type == "throughput":
            # Extract throughput values
            throughputs = []
            success_rates = []
            
            for api in apis:
                result = api_results[api]
                if result.get("status") == "success" and "summary" in result:
                    throughputs.append(result["summary"].get("requests_per_second", 0))
                    success_rates.append(result["summary"].get("success_rate", 0) * 100)  # To percentage
                else:
                    throughputs.append(0)
                    success_rates.append(0)
            
            # Create bar chart for throughput
            axes[0].bar(apis, throughputs)
            axes[0].set_title('API Throughput Comparison')
            axes[0].set_ylabel('Requests per Second')
            axes[0].grid(True, alpha=0.3)
            
            # Create bar chart for success rate
            axes[1].bar(apis, success_rates)
            axes[1].set_title('API Success Rate')
            axes[1].set_ylabel('Success Rate (%)')
            axes[1].set_ylim(0, 100)
            axes[1].yaxis.set_major_formatter(PercentFormatter())
            axes[1].grid(True, alpha=0.3)
            
        elif test_type == "reliability":
            # Extract reliability values
            success_rates = []
            avg_latencies = []
            
            for api in apis:
                result = api_results[api]
                if result.get("status") == "success" and "summary" in result:
                    success_rates.append(result["summary"].get("success_rate", 0) * 100)  # To percentage
                    avg_latencies.append(result["summary"].get("avg_latency", 0))
                else:
                    success_rates.append(0)
                    avg_latencies.append(0)
            
            # Create bar chart for success rate
            axes[0].bar(apis, success_rates)
            axes[0].set_title('API Reliability Comparison')
            axes[0].set_ylabel('Success Rate (%)')
            axes[0].set_ylim(0, 100)
            axes[0].yaxis.set_major_formatter(PercentFormatter())
            axes[0].grid(True, alpha=0.3)
            
            # Create bar chart for latency
            axes[1].bar(apis, avg_latencies)
            axes[1].set_title('API Average Latency')
            axes[1].set_ylabel('Latency (seconds)')
            axes[1].grid(True, alpha=0.3)
        
        # Finalize figure
        fig.suptitle(f'API Comparison - {test_type.capitalize()} Test', fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save figure
        if output_file:
            output_path = output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"api_comparison_{test_type}_{timestamp}.png"
        
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        logger.info(f"Saved comparison visualization to {output_path}")
        return output_path
    
    def _generate_cost_optimization_visualization(
        self,
        optimization_results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate visualization for cost optimization.
        
        Args:
            optimization_results: Results from run_cost_optimized_tests
            output_file: File to save visualization to
            
        Returns:
            Path to saved visualization
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot cost breakdown
        costs = optimization_results["costs"]
        cost_labels = list(costs.keys())
        cost_values = list(costs.values())
        
        # Sort by cost (descending)
        sorted_indices = np.argsort(cost_values)[::-1]
        sorted_labels = [cost_labels[i] for i in sorted_indices]
        sorted_values = [cost_values[i] for i in sorted_indices]
        
        axes[0, 0].bar(sorted_labels, sorted_values)
        axes[0, 0].set_title('Cost Breakdown by API and Test Type')
        axes[0, 0].set_ylabel('Cost ($)')
        axes[0, 0].set_xticklabels(sorted_labels, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot budget utilization
        total_cost = optimization_results["total_cost"]
        remaining_budget = optimization_results["remaining_budget"]
        budget = optimization_results["budget"]
        
        axes[0, 1].pie(
            [total_cost, remaining_budget],
            labels=['Used', 'Remaining'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#ff9999', '#66b3ff']
        )
        axes[0, 1].set_title(f'Budget Utilization (${budget:.2f} budget)')
        
        # Plot performance metrics if available
        test_types = optimization_results["test_types"]
        apis_tested = optimization_results["apis_tested"]
        
        if "latency" in test_types:
            latency_data = {}
            for api in apis_tested:
                if ("latency" in optimization_results["results"] and 
                    api in optimization_results["results"]["latency"] and
                    optimization_results["results"]["latency"][api].get("status") == "success"):
                    
                    result = optimization_results["results"]["latency"][api]
                    if "summary" in result and "avg_latency" in result["summary"]:
                        latency_data[api] = result["summary"]["avg_latency"]
            
            if latency_data:
                apis = list(latency_data.keys())
                values = list(latency_data.values())
                
                axes[1, 0].bar(apis, values)
                axes[1, 0].set_title('Latency Comparison')
                axes[1, 0].set_ylabel('Average Latency (s)')
                axes[1, 0].grid(True, alpha=0.3)
        
        if "throughput" in test_types:
            throughput_data = {}
            for api in apis_tested:
                if ("throughput" in optimization_results["results"] and 
                    api in optimization_results["results"]["throughput"] and
                    optimization_results["results"]["throughput"][api].get("status") == "success"):
                    
                    result = optimization_results["results"]["throughput"][api]
                    if "summary" in result and "requests_per_second" in result["summary"]:
                        throughput_data[api] = result["summary"]["requests_per_second"]
            
            if throughput_data:
                apis = list(throughput_data.keys())
                values = list(throughput_data.values())
                
                axes[1, 1].bar(apis, values)
                axes[1, 1].set_title('Throughput Comparison')
                axes[1, 1].set_ylabel('Requests per Second')
                axes[1, 1].grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle('Cost-Optimized API Testing Results', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        # Save figure
        if output_file:
            output_path = output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"cost_optimization_{timestamp}.png"
        
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        logger.info(f"Saved cost optimization visualization to {output_path}")
        return output_path
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.
        
        Returns:
            Dictionary with status information
        """
        # Get framework status
        api_framework_status = self.api_testing.get_api_framework_status()
        
        # Get scheduler status
        api_status = self.scheduler.get_api_status()
        scheduling_metrics = self.scheduler.get_scheduling_metrics()
        cost_report = self.scheduler.get_cost_report()
        
        # Get historical metrics summary
        historical_metrics = {
            metric: {
                api: len(data) for api, data in api_data.items()
            }
            for metric, api_data in self.historical_results["metrics"].items()
        }
        
        # Create comprehensive status
        status = {
            "status": "running",
            "session_id": self.current_session_id,
            "session_duration": time.time() - self.session_start_time,
            "api_framework": api_framework_status,
            "api_status": api_status,
            "scheduling": scheduling_metrics,
            "costs": cost_report,
            "historical": {
                "sessions": len(self.historical_results["sessions"]),
                "tests": len(self.historical_results["tests"]),
                "metrics": historical_metrics
            }
        }
        
        return status


def format_status_output(status: Dict[str, Any]) -> str:
    """Format status information for display."""
    output = []
    
    output.append("Advanced API Testing Framework Status")
    output.append("=" * 40)
    output.append(f"Status: {status['status']}")
    output.append(f"Session ID: {status['session_id']}")
    
    # Format duration
    duration = status["session_duration"]
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    output.append(f"Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # API Framework
    output.append("\nAPI Framework:")
    api_framework = status["api_framework"]
    output.append(f"  Available APIs: {', '.join(api_framework['available_api_types'])}")
    output.append(f"  Registered APIs: {', '.join(api_framework['registered_api_types'])}")
    
    # API Status
    output.append("\nAPI Status:")
    for api, api_status in status["api_status"].items():
        if isinstance(api_status, dict):
            circuit_status = api_status.get("circuit_breaker_status", "unknown")
            success_rate = api_status.get("success_rate", 0)
            performance = api_status.get("performance_score", 0)
            output.append(f"  {api}: {circuit_status}, {success_rate:.2%} success, {performance:.2f} performance")
    
    # Costs
    output.append("\nCost Report:")
    costs = status["costs"]
    output.append(f"  Total cost: ${costs['total_cost']:.4f}")
    for api, cost in costs['api_costs'].items():
        output.append(f"  {api}: ${cost:.4f}")
    
    # Historical
    output.append("\nHistorical Data:")
    historical = status["historical"]
    output.append(f"  Sessions: {historical['sessions']}")
    output.append(f"  Tests: {historical['tests']}")
    
    output.append("  Metrics:")
    for metric, api_counts in historical['metrics'].items():
        if api_counts:
            output.append(f"    {metric}: {sum(api_counts.values())} data points from {len(api_counts)} APIs")
    
    return "\n".join(output)


def load_api_keys_from_env():
    """Load API keys from environment variables."""
    api_keys = {}
    
    # Try to load from dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Look for common API key environment variables
    api_key_vars = [
        # OpenAI
        ("openai_api_key", "OPENAI_API_KEY"),
        ("openai_api_key_1", "OPENAI_API_KEY_1"),
        ("openai_api_key_2", "OPENAI_API_KEY_2"),
        
        # Claude
        ("claude_api_key", "CLAUDE_API_KEY"),
        ("claude_api_key", "ANTHROPIC_API_KEY"),
        ("claude_api_key_1", "ANTHROPIC_API_KEY_1"),
        
        # Groq
        ("groq_api_key", "GROQ_API_KEY"),
        ("groq_api_key_1", "GROQ_API_KEY_1"),
        
        # Gemini
        ("gemini_api_key", "GEMINI_API_KEY"),
        ("gemini_api_key", "GOOGLE_API_KEY")
    ]
    
    for key_name, env_var in api_key_vars:
        value = os.environ.get(env_var)
        if value:
            api_keys[key_name] = value
    
    return api_keys


def setup_argument_parser():
    """Set up and return the argument parser for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Advanced API Distributed Testing Framework Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a benchmark on all available APIs
  python run_advanced_api_tests.py benchmark --all-apis
  
  # Compare specific APIs on latency
  python run_advanced_api_tests.py compare --apis openai,claude --test latency --iterations 10
  
  # Run cost-optimized testing
  python run_advanced_api_tests.py optimize --budget 5.0 --tests latency,throughput
  
  # Visualize historical performance
  python run_advanced_api_tests.py visualize --metrics latency,throughput --days 30
  
  # Show framework status
  python run_advanced_api_tests.py status
"""
    )
    
    # Add command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run comprehensive benchmarks")
    benchmark_parser.add_argument("--apis", type=str, help="Comma-separated list of APIs to benchmark")
    benchmark_parser.add_argument("--tests", type=str, help="Comma-separated list of tests to run")
    benchmark_parser.add_argument("--all-apis", action="store_true", help="Include all available APIs")
    benchmark_parser.add_argument("--all-tests", action="store_true", help="Include all test types")
    benchmark_parser.add_argument("--prompt", type=str, default="Explain the concept of machine learning to a high school student.", 
                              help="Prompt to use for testing")
    benchmark_parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for tests")
    benchmark_parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare specific APIs")
    compare_parser.add_argument("--apis", type=str, required=True, help="Comma-separated list of APIs to compare")
    compare_parser.add_argument("--test", type=str, required=True, choices=[t.value for t in APITestType], 
                            help="Test type to run")
    compare_parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms.", 
                            help="Prompt to use for testing")
    compare_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for tests")
    compare_parser.add_argument("--concurrency", type=int, default=3, help="Concurrency level for throughput tests")
    compare_parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    compare_parser.add_argument("--output", type=str, help="Output file for results and visualizations")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Run cost-optimized tests")
    optimize_parser.add_argument("--budget", type=float, required=True, help="Maximum budget for testing in dollars")
    optimize_parser.add_argument("--tests", type=str, required=True, 
                             help="Comma-separated list of tests to run")
    optimize_parser.add_argument("--prompt", type=str, default="Write a short story about artificial intelligence.", 
                             help="Prompt to use for testing")
    optimize_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for tests")
    optimize_parser.add_argument("--concurrency", type=int, default=2, help="Concurrency level for throughput tests")
    optimize_parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    optimize_parser.add_argument("--output", type=str, help="Output file for results and visualizations")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize historical performance")
    visualize_parser.add_argument("--metrics", type=str, required=True, 
                              help="Comma-separated list of metrics to visualize (latency,throughput,reliability,cost)")
    visualize_parser.add_argument("--days", type=int, default=30, help="Number of days to look back")
    visualize_parser.add_argument("--apis", type=str, help="Comma-separated list of APIs to include")
    visualize_parser.add_argument("--output", type=str, help="Output file for visualizations")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show framework status")
    status_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available APIs")
    
    # Global options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--database", type=str, default="api_test_results.json", 
                      help="Path to results database file")
    parser.add_argument("--prometheus-port", type=int, default=8005, help="Port for Prometheus metrics")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    return parser


def main():
    """Main entry point for the script."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Default to status if no command provided
    if not args.command:
        args.command = "status"
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load API keys
    api_keys = load_api_keys_from_env()
    metadata = {"api_keys": api_keys}
    
    # Initialize the framework
    framework = AdvancedAPITestingFramework(
        prometheus_port=args.prometheus_port,
        database_path=args.database,
        config_file=args.config,
        metadata=metadata
    )
    
    try:
        # Execute appropriate command
        if args.command == TestCommand.BENCHMARK.value:
            # Parse APIs and tests
            api_types = None
            if args.apis:
                api_types = [api.strip() for api in args.apis.split(",")]
            
            test_types = None
            if args.tests:
                test_types = [test.strip() for test in args.tests.split(",")]
            
            # Set test parameters
            parameters = {
                "prompt": args.prompt,
                "iterations": args.iterations
            }
            
            # Run the benchmark
            results = framework.run_benchmark(
                api_types=api_types,
                test_types=test_types,
                parameters=parameters,
                include_all_apis=args.all_apis,
                include_all_tests=args.all_tests
            )
            
            # Output results
            if not args.quiet:
                print(f"Benchmark completed with {len(results['apis_tested'])} APIs and {len(results['test_types'])} test types")
                
                # Display summary if available
                if "summary" in results:
                    summary = results["summary"]
                    if "latency_ranking" in summary:
                        print(f"\nLatency Ranking: {', '.join(summary['latency_ranking'])}")
                    if "throughput_ranking" in summary:
                        print(f"Throughput Ranking: {', '.join(summary['throughput_ranking'])}")
                    if "reliability_ranking" in summary:
                        print(f"Reliability Ranking: {', '.join(summary['reliability_ranking'])}")
            
            # Save results if requested
            if args.output:
                output_path = args.output
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                if not args.quiet:
                    print(f"\nResults saved to: {output_path}")
            
        elif args.command == TestCommand.COMPARE.value:
            # Parse APIs
            api_types = [api.strip() for api in args.apis.split(",")]
            
            # Set test parameters
            parameters = {
                "prompt": args.prompt,
                "iterations": args.iterations,
                "concurrency": args.concurrency
            }
            
            # Run the comparison
            results = framework.compare_apis(
                api_types=api_types,
                test_type=args.test,
                parameters=parameters,
                visualize=args.visualize,
                output_file=args.output
            )
            
            # Output results
            if not args.quiet:
                print(f"API Comparison completed for {len(api_types)} APIs on {args.test} test")
                
                # Display summary if available
                if "summary" in results:
                    summary = results["summary"]
                    if args.test == "latency" and "fastest_api" in summary:
                        print(f"\nFastest API: {summary['fastest_api']}")
                        print(f"Latency Ranking: {', '.join(summary['latency_ranking'])}")
                    elif args.test == "throughput" and "highest_throughput_api" in summary:
                        print(f"\nHighest Throughput API: {summary['highest_throughput_api']}")
                        print(f"Throughput Ranking: {', '.join(summary['throughput_ranking'])}")
                    elif args.test == "reliability" and "most_reliable_api" in summary:
                        print(f"\nMost Reliable API: {summary['most_reliable_api']}")
                        print(f"Reliability Ranking: {', '.join(summary['reliability_ranking'])}")
                
                # Display visualization path if generated
                if args.visualize and "visualization_file" in results:
                    print(f"\nVisualization saved to: {results['visualization_file']}")
            
            # Save results if requested
            if args.output and args.output.endswith(".json"):
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                
                if not args.quiet:
                    print(f"\nResults saved to: {args.output}")
            
        elif args.command == TestCommand.OPTIMIZE.value:
            # Parse tests
            test_types = [test.strip() for test in args.tests.split(",")]
            
            # Set test parameters
            parameters = {
                "prompt": args.prompt,
                "iterations": args.iterations,
                "concurrency": args.concurrency
            }
            
            # Run cost-optimized tests
            results = framework.run_cost_optimized_tests(
                budget=args.budget,
                test_types=test_types,
                parameters=parameters,
                visualize=args.visualize,
                output_file=args.output
            )
            
            # Output results
            if not args.quiet:
                print(f"Cost-optimized tests completed with budget ${args.budget:.2f}")
                print(f"Tests run: {', '.join(test_types)}")
                print(f"APIs tested: {', '.join(results['apis_tested'])}")
                print(f"\nTotal cost: ${results['total_cost']:.4f}")
                print(f"Remaining budget: ${results['remaining_budget']:.4f}")
                print(f"Budget utilization: {results['budget_utilization']:.2%}")
                
                # Display visualization path if generated
                if args.visualize and "visualization_file" in results:
                    print(f"\nVisualization saved to: {results['visualization_file']}")
            
            # Save results if requested
            if args.output and args.output.endswith(".json"):
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                
                if not args.quiet:
                    print(f"\nResults saved to: {args.output}")
            
        elif args.command == TestCommand.VISUALIZE.value:
            # Parse metrics
            metrics = [metric.strip() for metric in args.metrics.split(",")]
            
            # Parse APIs if provided
            api_types = None
            if args.apis:
                api_types = [api.strip() for api in args.apis.split(",")]
            
            # Create visualization
            results = framework.visualize_historical_performance(
                metrics=metrics,
                days=args.days,
                api_types=api_types,
                output_file=args.output
            )
            
            # Output results
            if not args.quiet:
                if results.get("status") == "success":
                    print(f"Historical visualization created for {', '.join(metrics)} metrics")
                    print(f"Visualization saved to: {results['visualization_file']}")
                else:
                    print(f"Error creating visualization: {results.get('error', 'Unknown error')}")
            
        elif args.command == TestCommand.STATUS.value:
            # Get status
            status = framework.get_status()
            
            # Output status
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(format_status_output(status))
            
        elif args.command == TestCommand.LIST.value:
            # Get API info
            api_info = framework.api_testing.get_api_backend_info()
            
            # Output available APIs
            print("Available API Types:")
            for api_type in api_info["available_api_types"]:
                print(f"  {api_type}")
            
            # Output API capabilities if available
            if "api_backends" in api_info:
                print("\nAPI Capabilities:")
                for api_type, info in api_info["api_backends"].items():
                    capabilities = info.get("capabilities", [])
                    if capabilities:
                        print(f"  {api_type}: {', '.join(capabilities)}")
                    else:
                        print(f"  {api_type}: No capability information")
        
    finally:
        # Ensure framework is stopped
        framework.stop()


if __name__ == "__main__":
    main()