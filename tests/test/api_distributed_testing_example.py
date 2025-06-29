#!/usr/bin/env python
"""
API Distributed Testing Framework Example

This script demonstrates the full capabilities of the API Distributed Testing Framework
through a comprehensive example that:

1. Sets up the testing environment with multiple API backends
2. Runs basic performance tests (latency, throughput, reliability)
3. Performs API comparison with visualizations
4. Executes cost-optimized testing with budget management
5. Analyzes historical performance data
6. Integrates with the distributed testing framework for scalability

Usage:
    python api_distributed_testing_example.py [--demo DEMO_TYPE] [--output-dir OUTPUT_DIR]
    
    DEMO_TYPE options:
    - basic: Run basic API tests
    - comparison: Compare multiple APIs
    - cost: Run cost-optimized testing
    - distributed: Demonstrate distributed testing
    - visualization: Generate performance visualizations
    - all: Run all demos (default)
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import threading
from datetime import datetime

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API distributed testing components
try:
    from test_api_distributed_integration import APIDistributedTesting, APITestType
    from api_backend_distributed_scheduler import (
        APIBackendScheduler, APIRateLimitStrategy, APICostStrategy,
        APIBackendProfile, APIRateLimit, APICostProfile, APIPerformanceProfile
    )
except ImportError as e:
    print(f"Error importing API distributed testing components: {e}")
    print("Make sure you have implemented the API distributed testing modules.")
    sys.exit(1)

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization libraries not available. Install matplotlib, pandas, numpy, and seaborn for visualizations.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_testing_example")


class APITestingExample:
    """Example class demonstrating API Distributed Testing Framework features."""
    
    def __init__(self, output_dir: str = "api_test_results"):
        """
        Initialize the API Testing Example.
        
        Args:
            output_dir: Directory to save test results and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load API keys from environment
        api_keys = self._load_api_keys_from_env()
        
        # Initialize the testing framework
        self.api_testing = APIDistributedTesting(
            resources={},
            metadata={"api_keys": api_keys}
        )
        
        # Initialize the API backend scheduler
        self.scheduler = APIBackendScheduler(
            rate_limit_strategy=APIRateLimitStrategy.ADAPTIVE,
            cost_strategy=APICostStrategy.BALANCED
        )
        
        # Start the scheduler
        self.scheduler.start()
        
        # Track test results
        self.test_results = {}
        
        logger.info(f"API Testing Example initialized, output directory: {output_dir}")
    
    def _load_api_keys_from_env(self) -> Dict[str, str]:
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
            
            # Claude
            ("claude_api_key", "CLAUDE_API_KEY"),
            ("claude_api_key", "ANTHROPIC_API_KEY"),
            
            # Groq
            ("groq_api_key", "GROQ_API_KEY"),
            
            # Gemini
            ("gemini_api_key", "GEMINI_API_KEY"),
            ("gemini_api_key", "GOOGLE_API_KEY")
        ]
        
        for key_name, env_var in api_key_vars:
            value = os.environ.get(env_var)
            if value:
                api_keys[key_name] = value
                logger.debug(f"Found API key for {key_name}")
        
        if not api_keys:
            logger.warning("No API keys found in environment variables.")
        
        return api_keys
    
    def run_basic_tests(self) -> Dict[str, Any]:
        """
        Run basic API tests to demonstrate the framework.
        
        Returns:
            Dictionary of test results
        """
        logger.info("Running basic API tests...")
        
        # Get available API types
        api_info = self.api_testing.get_api_backend_info()
        available_apis = api_info["available_api_types"]
        
        if not available_apis:
            logger.warning("No API backends available for testing.")
            return {"status": "error", "message": "No API backends available"}
        
        logger.info(f"Available APIs: {', '.join(available_apis)}")
        
        # Choose the first available API for testing
        api_type = available_apis[0]
        logger.info(f"Testing with API: {api_type}")
        
        # Register the API backend
        self.api_testing.register_api_backend(api_type)
        
        # Update scheduler with API
        self.scheduler.register_api_backend(api_type)
        
        # Initialize results
        results = {
            "api_type": api_type,
            "timestamp": time.time(),
            "tests": {}
        }
        
        # Run a latency test
        logger.info("Running latency test...")
        latency_result = self.api_testing.run_distributed_test(
            api_type=api_type,
            test_type=APITestType.LATENCY,
            parameters={
                "prompt": "Explain the concept of distributed computing in 3 paragraphs.",
                "iterations": 3
            }
        )
        results["tests"]["latency"] = latency_result
        
        # Run a throughput test
        logger.info("Running throughput test...")
        throughput_result = self.api_testing.run_distributed_test(
            api_type=api_type,
            test_type=APITestType.THROUGHPUT,
            parameters={
                "prompt": "Write a one paragraph summary of quantum computing.",
                "concurrency": 2,
                "duration": 5
            }
        )
        results["tests"]["throughput"] = throughput_result
        
        # Run a reliability test
        logger.info("Running reliability test...")
        reliability_result = self.api_testing.run_distributed_test(
            api_type=api_type,
            test_type=APITestType.RELIABILITY,
            parameters={
                "prompt": "What is machine learning?",
                "iterations": 5,
                "interval": 1
            }
        )
        results["tests"]["reliability"] = reliability_result
        
        # Save results
        output_file = os.path.join(self.output_dir, "basic_tests.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Basic tests completed, results saved to {output_file}")
        
        # Print summary
        self._print_basic_results_summary(results)
        
        return results
    
    def _print_basic_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of basic test results."""
        api_type = results["api_type"]
        print(f"\nAPI Testing Results for {api_type}")
        print("=" * 40)
        
        # Latency test summary
        if "latency" in results["tests"]:
            latency = results["tests"]["latency"]
            if "summary" in latency and latency.get("status") == "success":
                summary = latency["summary"]
                print(f"Latency Test:")
                print(f"  Avg Latency: {summary.get('avg_latency', 'N/A'):.3f}s")
                print(f"  Min Latency: {summary.get('min_latency', 'N/A'):.3f}s")
                print(f"  Max Latency: {summary.get('max_latency', 'N/A'):.3f}s")
                print(f"  Success Rate: {summary.get('successful_iterations', 0)}/{latency.get('iterations', 0)}")
            else:
                print(f"Latency Test: Failed or incomplete")
        
        # Throughput test summary
        if "throughput" in results["tests"]:
            throughput = results["tests"]["throughput"]
            if "summary" in throughput and throughput.get("status") == "success":
                summary = throughput["summary"]
                print(f"\nThroughput Test:")
                print(f"  Requests/Second: {summary.get('requests_per_second', 'N/A'):.2f}")
                print(f"  Total Requests: {summary.get('total_requests', 'N/A')}")
                print(f"  Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                print(f"  Duration: {summary.get('actual_duration', 'N/A'):.2f}s")
            else:
                print(f"\nThroughput Test: Failed or incomplete")
        
        # Reliability test summary
        if "reliability" in results["tests"]:
            reliability = results["tests"]["reliability"]
            if "summary" in reliability and reliability.get("status") == "success":
                summary = reliability["summary"]
                print(f"\nReliability Test:")
                print(f"  Success Rate: {summary.get('success_rate', 'N/A'):.2%}")
                print(f"  Avg Latency: {summary.get('avg_latency', 'N/A'):.3f}s")
                print(f"  Successful: {summary.get('successful_iterations', 0)}/{reliability.get('iterations', 0)}")
            else:
                print(f"\nReliability Test: Failed or incomplete")
    
    def run_api_comparison(self) -> Dict[str, Any]:
        """
        Run API comparison to demonstrate the comparison capabilities.
        
        Returns:
            Dictionary of comparison results
        """
        logger.info("Running API comparison...")
        
        # Get available API types
        api_info = self.api_testing.get_api_backend_info()
        available_apis = api_info["available_api_types"]
        
        if not available_apis:
            logger.warning("No API backends available for comparison.")
            return {"status": "error", "message": "No API backends available"}
        
        # Need at least 2 APIs for comparison
        if len(available_apis) < 2:
            available_apis = [available_apis[0], available_apis[0]]  # Compare with itself as a fallback
            logger.warning(f"Only one API available ({available_apis[0]}), comparison will be limited.")
        
        # Use up to 3 APIs for comparison
        apis_to_compare = available_apis[:3]
        logger.info(f"Comparing APIs: {', '.join(apis_to_compare)}")
        
        # Register the API backends
        for api_type in apis_to_compare:
            self.api_testing.register_api_backend(api_type)
            self.scheduler.register_api_backend(api_type)
        
        # Initialize results
        results = {
            "apis_compared": apis_to_compare,
            "timestamp": time.time(),
            "comparisons": {}
        }
        
        # Compare APIs on latency
        logger.info("Comparing APIs on latency...")
        latency_comparison = self.api_testing.compare_apis(
            api_types=apis_to_compare,
            test_type="latency",
            parameters={
                "prompt": "Explain the advantages and disadvantages of using large language models.",
                "iterations": 3
            }
        )
        results["comparisons"]["latency"] = latency_comparison
        
        # Compare APIs on throughput
        logger.info("Comparing APIs on throughput...")
        throughput_comparison = self.api_testing.compare_apis(
            api_types=apis_to_compare,
            test_type="throughput",
            parameters={
                "prompt": "What are the main challenges in artificial intelligence safety?",
                "concurrency": 2,
                "duration": 5
            }
        )
        results["comparisons"]["throughput"] = throughput_comparison
        
        # Generate visualizations if available
        if VISUALIZATION_AVAILABLE:
            logger.info("Generating comparison visualizations...")
            
            # Generate latency visualization
            latency_vis_file = self._generate_comparison_visualization(
                latency_comparison,
                "latency",
                os.path.join(self.output_dir, "latency_comparison.png")
            )
            results["visualizations"] = {"latency": latency_vis_file}
            
            # Generate throughput visualization
            throughput_vis_file = self._generate_comparison_visualization(
                throughput_comparison,
                "throughput",
                os.path.join(self.output_dir, "throughput_comparison.png")
            )
            results["visualizations"]["throughput"] = throughput_vis_file
        
        # Save results
        output_file = os.path.join(self.output_dir, "api_comparison.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"API comparison completed, results saved to {output_file}")
        
        # Print summary
        self._print_comparison_results_summary(results)
        
        return results
    
    def _generate_comparison_visualization(
        self,
        comparison_results: Dict[str, Any],
        test_type: str,
        output_file: str
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
        if not VISUALIZATION_AVAILABLE:
            return "Visualization libraries not available"
        
        # Extract data for visualization
        api_results = comparison_results.get("results_by_api", {})
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
                result = api_results.get(api, {})
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
                result = api_results.get(api, {})
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
                result = api_results.get(api, {})
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
            axes[1].grid(True, alpha=0.3)
        
        # Finalize figure
        fig.suptitle(f'API Comparison - {test_type.capitalize()} Test', fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save figure
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        
        logger.info(f"Saved comparison visualization to {output_file}")
        return output_file
    
    def _print_comparison_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of API comparison results."""
        apis = results["apis_compared"]
        print(f"\nAPI Comparison Results for {', '.join(apis)}")
        print("=" * 40)
        
        # Latency comparison summary
        if "latency" in results["comparisons"]:
            latency_comp = results["comparisons"]["latency"]
            if "summary" in latency_comp:
                summary = latency_comp["summary"]
                if "latency_ranking" in summary:
                    print(f"Latency Ranking: {', '.join(summary.get('latency_ranking', []))}")
                    if "fastest_api" in summary:
                        print(f"Fastest API: {summary.get('fastest_api', 'N/A')}")
                
                # Print individual latencies
                print("\nLatency by API:")
                for api in apis:
                    if api in latency_comp.get("results_by_api", {}):
                        api_result = latency_comp["results_by_api"][api]
                        if "summary" in api_result and api_result.get("status") == "success":
                            api_summary = api_result["summary"]
                            print(f"  {api}: {api_summary.get('avg_latency', 'N/A'):.3f}s avg")
            else:
                print(f"Latency Comparison: Failed or incomplete")
        
        # Throughput comparison summary
        if "throughput" in results["comparisons"]:
            throughput_comp = results["comparisons"]["throughput"]
            if "summary" in throughput_comp:
                summary = throughput_comp["summary"]
                if "throughput_ranking" in summary:
                    print(f"\nThroughput Ranking: {', '.join(summary.get('throughput_ranking', []))}")
                    if "highest_throughput_api" in summary:
                        print(f"Highest Throughput API: {summary.get('highest_throughput_api', 'N/A')}")
                
                # Print individual throughputs
                print("\nThroughput by API:")
                for api in apis:
                    if api in throughput_comp.get("results_by_api", {}):
                        api_result = throughput_comp["results_by_api"][api]
                        if "summary" in api_result and api_result.get("status") == "success":
                            api_summary = api_result["summary"]
                            print(f"  {api}: {api_summary.get('requests_per_second', 'N/A'):.2f} req/s")
            else:
                print(f"\nThroughput Comparison: Failed or incomplete")
        
        # Print visualization paths if available
        if "visualizations" in results:
            print("\nVisualizations:")
            for test_type, path in results["visualizations"].items():
                print(f"  {test_type.capitalize()}: {path}")
    
    def run_cost_optimized_testing(self, budget: float = 1.0) -> Dict[str, Any]:
        """
        Run cost-optimized API testing to demonstrate budget management.
        
        Args:
            budget: Maximum budget for testing in dollars
            
        Returns:
            Dictionary of cost-optimized test results
        """
        logger.info(f"Running cost-optimized testing with budget ${budget:.2f}...")
        
        # Get available API types
        api_info = self.api_testing.get_api_backend_info()
        available_apis = api_info["available_api_types"]
        
        if not available_apis:
            logger.warning("No API backends available for testing.")
            return {"status": "error", "message": "No API backends available"}
        
        logger.info(f"Available APIs: {', '.join(available_apis)}")
        
        # Register all available APIs
        for api_type in available_apis:
            self.api_testing.register_api_backend(api_type)
            self.scheduler.register_api_backend(api_type)
            
            # Mock update cost profiles for demonstration
            if hasattr(self.scheduler, "api_profiles") and api_type in self.scheduler.api_profiles:
                profile = self.scheduler.api_profiles[api_type]
                
                # Set some example costs based on API type
                if api_type == "openai":
                    profile.cost_profile.cost_per_1k_input_tokens = 0.0005
                    profile.cost_profile.cost_per_1k_output_tokens = 0.0015
                elif api_type == "claude":
                    profile.cost_profile.cost_per_1k_input_tokens = 0.0030
                    profile.cost_profile.cost_per_1k_output_tokens = 0.0150
                elif api_type == "groq":
                    profile.cost_profile.cost_per_1k_input_tokens = 0.0001
                    profile.cost_profile.cost_per_1k_output_tokens = 0.0003
                else:
                    profile.cost_profile.cost_per_1k_input_tokens = 0.0010
                    profile.cost_profile.cost_per_1k_output_tokens = 0.0030
        
        # Configure scheduler for cost optimization
        self.scheduler.cost_strategy = APICostStrategy.MINIMIZE_COST
        
        # Initialize results
        results = {
            "timestamp": time.time(),
            "budget": budget,
            "apis_tested": [],
            "tests": {},
            "costs": {},
            "total_cost": 0.0
        }
        
        # Define test types
        test_types = ["latency", "throughput"]
        
        # Run tests while tracking costs
        remaining_budget = budget
        
        for test_type in test_types:
            results["tests"][test_type] = {}
            
            for api_type in available_apis:
                # Skip if budget is exhausted
                if remaining_budget <= 0.01:  # 1 cent minimum
                    logger.info(f"Budget exhausted, skipping {api_type} for {test_type}")
                    continue
                
                # Estimate cost for this test
                estimated_cost = self._estimate_test_cost(api_type, test_type)
                
                # Skip if estimated cost exceeds remaining budget
                if estimated_cost > remaining_budget:
                    logger.info(f"Estimated cost ${estimated_cost:.2f} exceeds remaining budget ${remaining_budget:.2f}")
                    continue
                
                logger.info(f"Running {test_type} test for {api_type} (est. cost: ${estimated_cost:.2f})")
                
                try:
                    # Parameters based on test type
                    if test_type == "latency":
                        parameters = {
                            "prompt": "Compare and contrast traditional machine learning with deep learning.",
                            "iterations": 2  # Reduced iterations for cost savings
                        }
                    else:  # throughput
                        parameters = {
                            "prompt": "Explain reinforcement learning briefly.",
                            "concurrency": 2,
                            "duration": 3  # Short duration for cost savings
                        }
                    
                    # Run the test
                    result = self.api_testing.run_distributed_test(
                        api_type=api_type,
                        test_type=test_type,
                        parameters=parameters
                    )
                    
                    # Record actual cost (in a real implementation, this would come from the API)
                    actual_cost = estimated_cost
                    
                    # Store the result
                    results["tests"][test_type][api_type] = result
                    results["costs"][f"{api_type}_{test_type}"] = actual_cost
                    results["total_cost"] += actual_cost
                    
                    # Add API to tested list if not already there
                    if api_type not in results["apis_tested"]:
                        results["apis_tested"].append(api_type)
                    
                    # Update remaining budget
                    remaining_budget -= actual_cost
                    
                except Exception as e:
                    logger.error(f"Error running {test_type} test for {api_type}: {e}")
                    results["tests"][test_type][api_type] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        # Add budget information
        results["remaining_budget"] = remaining_budget
        results["budget_utilization"] = (budget - remaining_budget) / budget if budget > 0 else 0
        
        # Generate visualization if available
        if VISUALIZATION_AVAILABLE:
            logger.info("Generating cost optimization visualization...")
            
            vis_file = self._generate_cost_optimization_visualization(
                results,
                os.path.join(self.output_dir, "cost_optimization.png")
            )
            results["visualization"] = vis_file
        
        # Save results
        output_file = os.path.join(self.output_dir, "cost_optimized_tests.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Cost-optimized testing completed, results saved to {output_file}")
        
        # Print summary
        self._print_cost_optimization_summary(results)
        
        return results
    
    def _estimate_test_cost(self, api_type: str, test_type: str) -> float:
        """
        Estimate the cost of a test based on typical usage.
        This is a simplified estimation for demonstration purposes.
        
        Args:
            api_type: Type of API
            test_type: Type of test
            
        Returns:
            Estimated cost in dollars
        """
        if not hasattr(self.scheduler, "api_profiles") or api_type not in self.scheduler.api_profiles:
            return 0.1  # Default cost estimate
        
        profile = self.scheduler.api_profiles[api_type]
        
        # Estimate based on test type
        if test_type == "latency":
            # 2 iterations, ~500 tokens each way
            input_tokens = 1000
            output_tokens = 1000
        elif test_type == "throughput":
            # ~5 requests, ~300 tokens each way
            input_tokens = 1500
            output_tokens = 1500
        else:
            input_tokens = 1000
            output_tokens = 1000
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * profile.cost_profile.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * profile.cost_profile.cost_per_1k_output_tokens
        
        return input_cost + output_cost
    
    def _generate_cost_optimization_visualization(self, results: Dict[str, Any], output_file: str) -> str:
        """
        Generate visualization for cost optimization results.
        
        Args:
            results: Cost optimization results
            output_file: File to save visualization to
            
        Returns:
            Path to saved visualization
        """
        if not VISUALIZATION_AVAILABLE:
            return "Visualization libraries not available"
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot cost breakdown
        costs = results["costs"]
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
        total_cost = results["total_cost"]
        remaining_budget = results["remaining_budget"]
        budget = results["budget"]
        
        axes[0, 1].pie(
            [total_cost, remaining_budget],
            labels=['Used', 'Remaining'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#ff9999', '#66b3ff']
        )
        axes[0, 1].set_title(f'Budget Utilization (${budget:.2f} budget)')
        
        # Plot performance metrics if available
        apis_tested = results["apis_tested"]
        
        # Latency comparison
        latency_data = {}
        for api in apis_tested:
            if ("latency" in results["tests"] and 
                api in results["tests"]["latency"] and
                results["tests"]["latency"][api].get("status") == "success"):
                
                result = results["tests"]["latency"][api]
                if "summary" in result and "avg_latency" in result["summary"]:
                    latency_data[api] = result["summary"]["avg_latency"]
        
        if latency_data:
            apis = list(latency_data.keys())
            values = list(latency_data.values())
            
            axes[1, 0].bar(apis, values)
            axes[1, 0].set_title('Latency Comparison')
            axes[1, 0].set_ylabel('Average Latency (s)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].set_title('Latency Comparison - No Data')
        
        # Throughput comparison
        throughput_data = {}
        for api in apis_tested:
            if ("throughput" in results["tests"] and 
                api in results["tests"]["throughput"] and
                results["tests"]["throughput"][api].get("status") == "success"):
                
                result = results["tests"]["throughput"][api]
                if "summary" in result and "requests_per_second" in result["summary"]:
                    throughput_data[api] = result["summary"]["requests_per_second"]
        
        if throughput_data:
            apis = list(throughput_data.keys())
            values = list(throughput_data.values())
            
            axes[1, 1].bar(apis, values)
            axes[1, 1].set_title('Throughput Comparison')
            axes[1, 1].set_ylabel('Requests per Second')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].set_title('Throughput Comparison - No Data')
        
        # Add overall title
        fig.suptitle('Cost-Optimized API Testing Results', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        # Save figure
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        
        logger.info(f"Saved cost optimization visualization to {output_file}")
        return output_file
    
    def _print_cost_optimization_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of cost optimization results."""
        print(f"\nCost-Optimized Testing Results (Budget: ${results['budget']:.2f})")
        print("=" * 40)
        
        # Cost summary
        print(f"Total Cost: ${results['total_cost']:.4f}")
        print(f"Remaining Budget: ${results['remaining_budget']:.4f}")
        print(f"Budget Utilization: {results['budget_utilization']:.2%}")
        
        # APIs tested
        print(f"\nAPIs Tested: {', '.join(results['apis_tested'])}")
        
        # Cost breakdown
        print("\nCost Breakdown:")
        for key, cost in results["costs"].items():
            print(f"  {key}: ${cost:.4f}")
        
        # Test results summary
        for test_type in results["tests"]:
            print(f"\n{test_type.capitalize()} Test Results:")
            for api, result in results["tests"][test_type].items():
                if result.get("status") == "success" and "summary" in result:
                    if test_type == "latency":
                        print(f"  {api}: {result['summary'].get('avg_latency', 'N/A'):.3f}s avg latency")
                    elif test_type == "throughput":
                        print(f"  {api}: {result['summary'].get('requests_per_second', 'N/A'):.2f} req/s")
                else:
                    print(f"  {api}: Failed or incomplete")
        
        # Visualization path
        if "visualization" in results:
            print(f"\nVisualization: {results['visualization']}")
    
    def run_distributed_testing_demo(self) -> Dict[str, Any]:
        """
        Demonstrate distributed testing capabilities.
        
        Returns:
            Dictionary of distributed testing results
        """
        logger.info("Running distributed testing demonstration...")
        
        # Get available API types
        api_info = self.api_testing.get_api_backend_info()
        available_apis = api_info["available_api_types"]
        
        if not available_apis:
            logger.warning("No API backends available for testing.")
            return {"status": "error", "message": "No API backends available"}
        
        # Choose an API for testing
        api_type = available_apis[0]
        logger.info(f"Using API: {api_type}")
        
        # Register the API backend
        self.api_testing.register_api_backend(api_type)
        self.scheduler.register_api_backend(api_type)
        
        # Create test tasks
        tasks = []
        for i in range(5):
            task_id = f"dist-test-task-{i}"
            prompt = f"Provide a {i+1} paragraph explanation of distributed systems."
            
            # Create a task with different parameters
            task = {
                "task_id": task_id,
                "api_type": api_type,
                "test_type": "latency" if i % 2 == 0 else "throughput",
                "parameters": {
                    "prompt": prompt,
                    "iterations": 2 if i % 2 == 0 else None,
                    "concurrency": None if i % 2 == 0 else 2,
                    "duration": None if i % 2 == 0 else 3
                },
                "priority": 5 - (i % 3)  # Different priorities
            }
            tasks.append(task)
        
        # Initialize results
        results = {
            "timestamp": time.time(),
            "api_type": api_type,
            "tasks": tasks,
            "task_results": {},
            "metrics": {}
        }
        
        # Create a thread pool to simulate distributed execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            futures = {}
            for task in tasks:
                task_id = task["task_id"]
                future = executor.submit(
                    self._execute_distributed_task,
                    task["api_type"],
                    task["test_type"],
                    task["parameters"],
                    task["priority"]
                )
                futures[future] = task_id
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    task_result = future.result()
                    results["task_results"][task_id] = task_result
                    logger.info(f"Task {task_id} completed successfully")
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    results["task_results"][task_id] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        # Get scheduler metrics
        results["metrics"] = self.scheduler.get_scheduling_metrics()
        
        # Save results
        output_file = os.path.join(self.output_dir, "distributed_testing.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Distributed testing demo completed, results saved to {output_file}")
        
        # Print summary
        self._print_distributed_testing_summary(results)
        
        return results
    
    def _execute_distributed_task(
        self,
        api_type: str,
        test_type: str,
        parameters: Dict[str, Any],
        priority: int
    ) -> Dict[str, Any]:
        """
        Execute a task as part of distributed testing.
        
        Args:
            api_type: API type to test
            test_type: Type of test to run
            parameters: Test parameters
            priority: Task priority
            
        Returns:
            Task result
        """
        # Add a small random delay to simulate network latency
        time.sleep(random.uniform(0.1, 0.5))
        
        # Run the test
        result = self.api_testing.run_distributed_test(
            api_type=api_type,
            test_type=test_type,
            parameters=parameters,
            priority=priority
        )
        
        return result
    
    def _print_distributed_testing_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of distributed testing results."""
        print(f"\nDistributed Testing Results")
        print("=" * 40)
        
        # Task summary
        print(f"API: {results['api_type']}")
        print(f"Total Tasks: {len(results['tasks'])}")
        
        # Task results
        completed = sum(1 for r in results["task_results"].values() if r.get("status") == "success")
        failed = len(results["task_results"]) - completed
        print(f"Tasks Completed: {completed}")
        print(f"Tasks Failed: {failed}")
        
        # Scheduler metrics
        if "metrics" in results:
            metrics = results["metrics"]
            print("\nScheduler Metrics:")
            if "tasks_scheduled" in metrics:
                print(f"  Tasks Scheduled: {metrics.get('tasks_scheduled', 0)}")
            if "tasks_completed" in metrics:
                print(f"  Tasks Completed: {metrics.get('tasks_completed', 0)}")
            if "pending_tasks" in metrics:
                print(f"  Pending Tasks: {metrics.get('pending_tasks', 0)}")
            if "running_tasks" in metrics:
                print(f"  Running Tasks: {metrics.get('running_tasks', 0)}")
            
            # API performance
            if "api_performance" in metrics:
                print("\nAPI Performance:")
                for api, perf in metrics["api_performance"].items():
                    avg_latency = perf.get("avg_latency", "N/A")
                    if avg_latency is not None and avg_latency != "N/A":
                        avg_latency = f"{avg_latency:.3f}s"
                    
                    success_rate = perf.get("success_rate", "N/A")
                    if success_rate is not None and success_rate != "N/A":
                        success_rate = f"{success_rate:.2%}"
                    
                    print(f"  {api}: {avg_latency} avg latency, {success_rate} success rate")
    
    def generate_performance_visualization(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate visualizations of performance metrics.
        
        Args:
            days: Number of days of data to simulate for visualization
            
        Returns:
            Dictionary with visualization results
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available.")
            return {"status": "error", "message": "Visualization libraries not available"}
        
        logger.info(f"Generating performance visualizations for {days} days of simulated data...")
        
        # Get available API types
        api_info = self.api_testing.get_api_backend_info()
        available_apis = api_info["available_api_types"]
        
        if not available_apis:
            logger.warning("No API backends available for visualization.")
            return {"status": "error", "message": "No API backends available"}
        
        # Generate simulated historical data
        historical_data = self._generate_simulated_historical_data(available_apis, days)
        
        # Initialize results
        results = {
            "timestamp": time.time(),
            "days_simulated": days,
            "apis": available_apis,
            "metrics": ["latency", "throughput", "reliability"],
            "visualizations": {}
        }
        
        # Generate visualizations for each metric
        for metric in results["metrics"]:
            vis_file = self._generate_historical_visualization(
                historical_data,
                metric,
                available_apis,
                os.path.join(self.output_dir, f"{metric}_history.png")
            )
            results["visualizations"][metric] = vis_file
        
        # Save results
        output_file = os.path.join(self.output_dir, "performance_visualization.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Performance visualization completed, results saved to {output_file}")
        
        # Print summary
        print(f"\nPerformance Visualization Results")
        print("=" * 40)
        print(f"Simulated data for {days} days")
        print(f"APIs included: {', '.join(available_apis)}")
        print(f"Metrics visualized: {', '.join(results['metrics'])}")
        print("\nVisualization files:")
        for metric, path in results["visualizations"].items():
            print(f"  {metric}: {path}")
        
        return results
    
    def _generate_simulated_historical_data(self, apis: List[str], days: int) -> Dict[str, Any]:
        """
        Generate simulated historical performance data for visualization.
        
        Args:
            apis: List of API types to generate data for
            days: Number of days of data to simulate
            
        Returns:
            Dictionary with simulated historical data
        """
        # Current timestamp as end date
        end_time = time.time()
        # Start date (days ago)
        start_time = end_time - (days * 24 * 60 * 60)
        
        # Generate data points (one per 6 hours)
        interval = 6 * 60 * 60  # 6 hours in seconds
        timestamps = []
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += interval
        
        # Initialize data structure
        data = {
            "latency": {},
            "throughput": {},
            "reliability": {}
        }
        
        # Generate data for each API and metric
        for api in apis:
            # Latency data (seconds) - generally getting better over time with some fluctuation
            base_latency = random.uniform(0.8, 3.0)  # Base latency varies by API
            improvement_factor = random.uniform(0.3, 0.6)  # How much it improves over time
            
            latency_data = []
            for i, ts in enumerate(timestamps):
                # Trend downward (improving) with random noise
                progress = i / max(1, len(timestamps) - 1)
                current_latency = base_latency * (1 - (progress * improvement_factor))
                # Add noise
                noise = random.uniform(-0.2, 0.3) * current_latency
                latency = max(0.1, current_latency + noise)
                
                latency_data.append({
                    "timestamp": ts,
                    "avg_latency": latency,
                    "min_latency": latency * 0.8,
                    "max_latency": latency * 1.4
                })
            
            data["latency"][api] = latency_data
            
            # Throughput data (requests per second) - improving over time
            base_throughput = random.uniform(1.0, 5.0)  # Base throughput varies by API
            improvement_factor = random.uniform(0.5, 1.2)  # How much it improves
            
            throughput_data = []
            for i, ts in enumerate(timestamps):
                # Trend upward (improving) with random noise
                progress = i / max(1, len(timestamps) - 1)
                current_throughput = base_throughput * (1 + (progress * improvement_factor))
                # Add noise
                noise = random.uniform(-0.1, 0.2) * current_throughput
                throughput = max(0.5, current_throughput + noise)
                
                throughput_data.append({
                    "timestamp": ts,
                    "requests_per_second": throughput
                })
            
            data["throughput"][api] = throughput_data
            
            # Reliability data (success rate) - high with occasional dips
            base_reliability = random.uniform(0.95, 0.99)  # Base reliability varies by API
            
            reliability_data = []
            for i, ts in enumerate(timestamps):
                # Generally high with occasional dips
                if random.random() < 0.1:  # 10% chance of a dip
                    success_rate = base_reliability * random.uniform(0.85, 0.95)
                else:
                    success_rate = base_reliability * random.uniform(0.98, 1.02)
                    success_rate = min(1.0, success_rate)  # Cap at 100%
                
                reliability_data.append({
                    "timestamp": ts,
                    "success_rate": success_rate
                })
            
            data["reliability"][api] = reliability_data
        
        return data
    
    def _generate_historical_visualization(
        self,
        historical_data: Dict[str, Any],
        metric: str,
        apis: List[str],
        output_file: str
    ) -> str:
        """
        Create visualization of historical performance data.
        
        Args:
            historical_data: Dictionary with historical data
            metric: Metric to visualize (latency, throughput, reliability)
            apis: List of APIs to include
            output_file: File to save visualization to
            
        Returns:
            Path to saved visualization
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create color map for APIs
        api_colors = {}
        colormap = plt.cm.tab10
        for i, api in enumerate(apis):
            api_colors[api] = colormap(i % 10)
        
        # Plot data for each API
        for api in apis:
            if api not in historical_data[metric]:
                continue
            
            # Get data points
            data = historical_data[metric][api]
            
            # Convert timestamps to datetime objects
            timestamps = [datetime.fromtimestamp(d["timestamp"]) for d in data]
            
            # Extract y values based on metric
            if metric == "latency":
                y_values = [d["avg_latency"] for d in data]
                y_label = "Average Latency (s)"
                title = "API Latency over Time"
                
                # Also plot min/max as a shaded area
                min_values = [d["min_latency"] for d in data]
                max_values = [d["max_latency"] for d in data]
                plt.fill_between(timestamps, min_values, max_values, alpha=0.2, color=api_colors[api])
                
            elif metric == "throughput":
                y_values = [d["requests_per_second"] for d in data]
                y_label = "Requests per Second"
                title = "API Throughput over Time"
                
            elif metric == "reliability":
                y_values = [d["success_rate"] * 100 for d in data]  # Convert to percentage
                y_label = "Success Rate (%)"
                title = "API Reliability over Time"
            
            # Plot the line
            plt.plot(timestamps, y_values, 'o-', label=api, color=api_colors[api])
        
        # Configure plot
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format date axis
        plt.gcf().autofmt_xdate()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logger.info(f"Saved {metric} visualization to {output_file}")
        return output_file
    
    def cleanup(self) -> None:
        """Clean up resources and save final results."""
        logger.info("Cleaning up resources...")
        
        # Stop the scheduler
        self.scheduler.stop()
        
        # Create summary of all tests
        summary = {
            "timestamp": time.time(),
            "output_directory": self.output_dir,
            "test_results": self.test_results
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")
        logger.info("Cleanup complete")


def run_demo(demo_type: str, output_dir: str) -> None:
    """
    Run specified demo type.
    
    Args:
        demo_type: Type of demo to run
        output_dir: Directory to save results
    """
    # Initialize the example
    example = APITestingExample(output_dir=output_dir)
    
    try:
        if demo_type == "basic" or demo_type == "all":
            print("\n=== Running Basic API Tests ===")
            basic_results = example.run_basic_tests()
            example.test_results["basic"] = basic_results
        
        if demo_type == "comparison" or demo_type == "all":
            print("\n=== Running API Comparison ===")
            comparison_results = example.run_api_comparison()
            example.test_results["comparison"] = comparison_results
        
        if demo_type == "cost" or demo_type == "all":
            print("\n=== Running Cost-Optimized Testing ===")
            cost_results = example.run_cost_optimized_testing(budget=1.0)
            example.test_results["cost"] = cost_results
        
        if demo_type == "distributed" or demo_type == "all":
            print("\n=== Running Distributed Testing Demo ===")
            distributed_results = example.run_distributed_testing_demo()
            example.test_results["distributed"] = distributed_results
        
        if demo_type == "visualization" or demo_type == "all":
            print("\n=== Generating Performance Visualizations ===")
            visualization_results = example.generate_performance_visualization(days=7)
            example.test_results["visualization"] = visualization_results
            
    finally:
        # Clean up resources
        example.cleanup()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="API Distributed Testing Framework Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--demo", choices=["basic", "comparison", "cost", "distributed", "visualization", "all"],
                      default="all", help="Type of demo to run")
    parser.add_argument("--output-dir", type=str, default="api_test_results",
                      help="Directory to save test results")
    
    args = parser.parse_args()
    
    try:
        run_demo(args.demo, args.output_dir)
        print(f"\nDemo completed successfully. Results saved to {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()