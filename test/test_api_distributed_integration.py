#!/usr/bin/env python
"""
API Backend Integration with Distributed Testing Framework

This module provides comprehensive integration between API backends and the
Distributed Testing Framework, enabling:

1. Unified testing interface for all API types
2. Comprehensive performance metrics collection for API benchmarking
3. Distributed execution of API tests across worker nodes
4. Anomaly detection for API performance metrics

Usage:
    # Create a distributed API testing framework
    api_testing = APIDistributedTesting()
    
    # Add API backends for testing
    api_testing.register_api_backend("openai")
    api_testing.register_api_backend("groq")
    
    # Run distributed tests
    results = api_testing.run_distributed_test(
        api_type="openai",
        test_type="latency",
        parameters={"prompt": "Hello, world!", "iterations": 10}
    )
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path
import importlib
import concurrent.futures
import tempfile
import uuid

# Add project root to the path to ensure imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import framework components
from test.distributed_testing.integration import DistributedTestingFramework, create_distributed_testing_framework
from test.distributed_testing.advanced_scheduling import Task, Worker
from test.distributed_testing.ml_anomaly_detection import MLAnomalyDetection
from test.distributed_testing.prometheus_grafana_integration import PrometheusGrafanaIntegration

# Import API backends
try:
    from ipfs_accelerate_py.api_backends import (
        openai_api, claude, groq, gemini, 
        hf_tgi, hf_tei, ollama, opea, 
        ovms, s3_kit, llvm
    )
except ImportError as e:
    print(f"Warning: Could not import some API backends: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_distributed_integration")


class APITestType(Enum):
    """Types of API tests that can be performed."""
    LATENCY = "latency"              # Measure response time
    THROUGHPUT = "throughput"        # Measure requests per second
    CONCURRENCY = "concurrency"      # Test concurrent requests
    RELIABILITY = "reliability"      # Test API reliability/uptime
    STREAMING = "streaming"          # Test streaming performance
    TOKEN_COST = "token_cost"        # Measure token usage/cost
    MODEL_COMPARISON = "model_comparison"  # Compare different models
    FAULT_TOLERANCE = "fault_tolerance"    # Test API fault tolerance


class APIDistributedTesting:
    """
    Distributed testing framework specifically for API backends.
    
    This class integrates API backends with the distributed testing framework,
    enabling comprehensive testing and benchmarking across distributed nodes.
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        coordinator_id: Optional[str] = None,
        prometheus_port: int = 8005,
        resources: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize the API distributed testing framework.
        
        Args:
            config_file: Optional path to a configuration file
            coordinator_id: Optional unique identifier for this coordinator
            prometheus_port: Port for Prometheus metrics
            resources: Resources dict for API backends
            metadata: Metadata dict for API backends (including API keys)
        """
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Create monitoring configuration
        monitoring_config = {
            "prometheus_port": prometheus_port,
            "metrics_collection_interval": 15,
            "anomaly_detection_interval": 60
        }
        
        # Create ML configuration
        ml_config = {
            "algorithms": ["isolation_forest", "threshold"],
            "forecasting": ["exponential_smoothing"],
            "visualization": True
        }
        
        # Create custom scheduler configuration
        scheduler_config = {
            "algorithm": "resource_aware",  # Use resource-aware scheduling for APIs
            "resource_match_weight": 0.8,   # Higher weight for resource matching
            "fairness_window": 50,          # Number of tasks to consider for fairness
            "adaptive_interval": 20         # Interval for adapting scheduling algorithm
        }
        
        # Initialize the distributed testing framework
        self.framework = create_distributed_testing_framework(
            config_file=config_file,
            coordinator_id=coordinator_id or f"api-coordinator-{uuid.uuid4().hex[:8]}",
            monitoring_config=monitoring_config,
            ml_config=ml_config,
            scheduler_config=scheduler_config
        )
        
        # Initialize API backend registry
        self.api_backends = {}
        self.api_instances = {}
        self.available_api_types = set()
        
        # Initialize the available API types
        self._discover_api_backends()
        
        # Register test metrics
        self.metrics = {
            "latency": {},
            "throughput": {},
            "reliability": {},
            "test_history": []
        }
        
        # Register worker capabilities for testing APIs
        self._register_local_worker()
        
        logger.info(f"API Distributed Testing initialized with {len(self.available_api_types)} API types")
    
    def _discover_api_backends(self):
        """Discover available API backends."""
        # Check for available API backends in ipfs_accelerate_py.api_backends
        api_module_pairs = [
            ("openai", openai_api),
            ("claude", claude),
            ("groq", groq),
            ("gemini", gemini),
            ("hf_tgi", hf_tgi),
            ("hf_tei", hf_tei),
            ("ollama", ollama),
            ("opea", opea),
            ("ovms", ovms),
            ("s3_kit", s3_kit),
            ("llvm", llvm)
        ]
        
        for api_name, module in api_module_pairs:
            if module is not None:
                self.available_api_types.add(api_name)
                self.api_backends[api_name] = module
                logger.debug(f"Discovered API backend: {api_name}")
    
    def _register_local_worker(self):
        """Register a local worker with API testing capabilities."""
        # Create a worker with API testing capabilities
        worker_data = {
            "worker_id": f"api-worker-{uuid.uuid4().hex[:8]}",
            "worker_type": "api_tester",
            "capabilities": {
                "api_testing": True,
                "supported_apis": list(self.available_api_types),
                "max_concurrent_tests": 10,
                "cpu": 4,
                "memory": 8
            },
            "status": "idle",
            "metadata": {
                "hostname": "localhost",
                "location": "local"
            }
        }
        
        # Register the worker with the framework
        worker_id = self.framework.add_worker(worker_data)
        logger.info(f"Registered local API testing worker with ID: {worker_id}")
        
        return worker_id
    
    def register_api_backend(self, api_type: str) -> bool:
        """
        Register an API backend for testing.
        
        Args:
            api_type: The type of API to register (e.g., "openai", "groq")
            
        Returns:
            True if registration was successful, False otherwise
        """
        if api_type not in self.available_api_types:
            logger.warning(f"API type '{api_type}' not available")
            return False
        
        try:
            # Initialize the API backend
            if api_type not in self.api_instances:
                # Get the module for this API type
                module = self.api_backends[api_type]
                
                # Get the class for this API type
                api_class = getattr(module, api_type)
                
                # Initialize the API instance
                api_instance = api_class(resources=self.resources, metadata=self.metadata)
                
                # Store the instance
                self.api_instances[api_type] = api_instance
                
                logger.info(f"Registered API backend: {api_type}")
                return True
            
            # Already registered
            return True
            
        except Exception as e:
            logger.error(f"Failed to register API backend '{api_type}': {e}")
            return False
    
    def run_distributed_test(
        self,
        api_type: str,
        test_type: Union[str, APITestType],
        parameters: Dict[str, Any],
        priority: int = 5
    ) -> Dict[str, Any]:
        """
        Run a distributed test for an API backend.
        
        Args:
            api_type: The type of API to test (e.g., "openai", "groq")
            test_type: The type of test to run (use APITestType enum)
            parameters: Test parameters
            priority: Test priority (higher = more important)
            
        Returns:
            Test results
        """
        # Check if API type is available
        if api_type not in self.available_api_types:
            logger.warning(f"API type '{api_type}' not available")
            return {"status": "error", "error": f"API type '{api_type}' not available"}
        
        # Register the API backend if not already registered
        if not self.register_api_backend(api_type):
            return {"status": "error", "error": f"Failed to register API backend '{api_type}'"}
        
        # Convert test_type to string if it's an enum
        if isinstance(test_type, APITestType):
            test_type = test_type.value
        
        # Create a task for this test
        task_data = {
            "task_id": f"api-test-{api_type}-{test_type}-{uuid.uuid4().hex[:8]}",
            "task_type": "api_test",
            "user_id": "api_testing",
            "priority": priority,
            "estimated_duration": parameters.get("estimated_duration", 60.0),
            "required_resources": {
                "api_testing": True,
                f"api_{api_type}": True
            },
            "metadata": {
                "api_type": api_type,
                "test_type": test_type,
                "parameters": parameters,
                "timestamp": time.time()
            }
        }
        
        # Add the task to the framework
        task_id = self.framework.add_task(task_data)
        if not task_id:
            logger.error(f"Failed to add test task for API '{api_type}'")
            return {"status": "error", "error": "Failed to add test task"}
        
        logger.info(f"Added API test task with ID: {task_id}")
        
        # In a real distributed system, we would now wait for the task to be scheduled
        # and executed on a worker node. For this implementation, we'll execute it locally.
        test_result = self._execute_api_test(api_type, test_type, parameters)
        
        # Update the task with the result
        worker_id = next(iter(self.framework.scheduler.workers.keys()), None)
        if worker_id:
            self.framework.complete_task(worker_id, success=True, result=test_result)
        
        # Record the test result
        self._record_test_result(api_type, test_type, parameters, test_result)
        
        return test_result
    
    def _execute_api_test(
        self, 
        api_type: str, 
        test_type: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an API test.
        
        Args:
            api_type: The type of API to test
            test_type: The type of test to run
            parameters: Test parameters
            
        Returns:
            Test results
        """
        logger.info(f"Executing {test_type} test for {api_type} API")
        
        # Get the API instance
        api_instance = self.api_instances.get(api_type)
        if not api_instance:
            return {"status": "error", "error": f"API instance '{api_type}' not found"}
        
        # Execute the appropriate test based on test_type
        test_method_name = f"_test_{test_type}"
        if hasattr(self, test_method_name):
            test_method = getattr(self, test_method_name)
            result = test_method(api_instance, parameters)
            return result
        else:
            logger.error(f"Unsupported test type: {test_type}")
            return {"status": "error", "error": f"Unsupported test type: {test_type}"}
    
    def _test_latency(self, api_instance, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test API latency by measuring response time.
        
        Args:
            api_instance: The API instance to test
            parameters: Test parameters
            
        Returns:
            Test results with latency metrics
        """
        # Extract parameters
        prompt = parameters.get("prompt", "Hello, world!")
        iterations = parameters.get("iterations", 5)
        model = parameters.get("model")
        
        results = {
            "status": "success",
            "test_type": "latency",
            "api_type": api_instance.__class__.__name__,
            "iterations": iterations,
            "latencies": [],
            "summary": {}
        }
        
        # Run the latency test
        for i in range(iterations):
            try:
                # Measure start time
                start_time = time.time()
                
                # Make API call based on API type
                if hasattr(api_instance, "completion"):
                    response = api_instance.completion(prompt, model=model)
                elif hasattr(api_instance, "chat_completion"):
                    response = api_instance.chat_completion([{"role": "user", "content": prompt}], model=model)
                else:
                    # Use the simplest API call we can find
                    # Use __test__ as a last resort
                    response = api_instance.__test__()
                
                # Calculate latency
                latency = time.time() - start_time
                
                # Record result
                results["latencies"].append(latency)
                
                logger.debug(f"Iteration {i+1}/{iterations}: Latency = {latency:.3f}s")
                
            except Exception as e:
                logger.error(f"Error in latency test iteration {i+1}: {e}")
                results["latencies"].append(None)
                results["errors"] = results.get("errors", []) + [str(e)]
        
        # Calculate summary statistics
        valid_latencies = [l for l in results["latencies"] if l is not None]
        if valid_latencies:
            results["summary"] = {
                "min_latency": min(valid_latencies),
                "max_latency": max(valid_latencies),
                "avg_latency": sum(valid_latencies) / len(valid_latencies),
                "successful_iterations": len(valid_latencies),
                "failed_iterations": iterations - len(valid_latencies)
            }
        else:
            results["status"] = "error"
            results["error"] = "All iterations failed"
        
        return results
    
    def _test_throughput(self, api_instance, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test API throughput by measuring requests per second.
        
        Args:
            api_instance: The API instance to test
            parameters: Test parameters
            
        Returns:
            Test results with throughput metrics
        """
        # Extract parameters
        prompt = parameters.get("prompt", "Hello, world!")
        duration = parameters.get("duration", 10)  # seconds
        concurrency = parameters.get("concurrency", 3)
        model = parameters.get("model")
        
        results = {
            "status": "success",
            "test_type": "throughput",
            "api_type": api_instance.__class__.__name__,
            "duration": duration,
            "concurrency": concurrency,
            "requests": [],
            "summary": {}
        }
        
        # Define a function to make API requests
        def make_request():
            try:
                start_time = time.time()
                
                # Make API call based on API type
                if hasattr(api_instance, "completion"):
                    response = api_instance.completion(prompt, model=model)
                elif hasattr(api_instance, "chat_completion"):
                    response = api_instance.chat_completion([{"role": "user", "content": prompt}], model=model)
                else:
                    # Use the simplest API call we can find
                    response = api_instance.__test__()
                
                latency = time.time() - start_time
                
                return {
                    "success": True,
                    "latency": latency,
                    "timestamp": start_time
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        # Run the throughput test
        start_time = time.time()
        end_time = start_time + duration
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit initial batch of requests
            futures = [executor.submit(make_request) for _ in range(concurrency)]
            
            # Continue submitting requests until duration is reached
            while time.time() < end_time:
                # Check for completed futures
                for future in concurrent.futures.as_completed(futures, timeout=0.1):
                    # Get the result
                    result = future.result()
                    results["requests"].append(result)
                    
                    # Submit a new request if time permits
                    if time.time() < end_time:
                        futures.append(executor.submit(make_request))
                
                # Short sleep to avoid CPU spinning
                time.sleep(0.01)
        
        # Wait for any remaining futures to complete
        for future in concurrent.futures.as_completed(futures):
            results["requests"].append(future.result())
        
        # Calculate summary statistics
        total_requests = len(results["requests"])
        successful_requests = sum(1 for r in results["requests"] if r.get("success", False))
        actual_duration = time.time() - start_time
        
        results["summary"] = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "actual_duration": actual_duration,
            "requests_per_second": successful_requests / actual_duration if actual_duration > 0 else 0,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0
        }
        
        return results
    
    def _test_reliability(self, api_instance, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test API reliability by checking success rate over multiple requests.
        
        Args:
            api_instance: The API instance to test
            parameters: Test parameters
            
        Returns:
            Test results with reliability metrics
        """
        # Extract parameters
        iterations = parameters.get("iterations", 10)
        interval = parameters.get("interval", 1)  # seconds between requests
        prompt = parameters.get("prompt", "Hello, world!")
        model = parameters.get("model")
        
        results = {
            "status": "success",
            "test_type": "reliability",
            "api_type": api_instance.__class__.__name__,
            "iterations": iterations,
            "interval": interval,
            "results": [],
            "summary": {}
        }
        
        # Run the reliability test
        for i in range(iterations):
            try:
                start_time = time.time()
                
                # Make API call based on API type
                if hasattr(api_instance, "completion"):
                    response = api_instance.completion(prompt, model=model)
                    success = True
                elif hasattr(api_instance, "chat_completion"):
                    response = api_instance.chat_completion([{"role": "user", "content": prompt}], model=model)
                    success = True
                else:
                    # Use the simplest API call we can find
                    response = api_instance.__test__()
                    success = True
                
                latency = time.time() - start_time
                
                results["results"].append({
                    "iteration": i + 1,
                    "success": success,
                    "latency": latency,
                    "timestamp": start_time
                })
                
            except Exception as e:
                results["results"].append({
                    "iteration": i + 1,
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                })
            
            # Wait for the interval
            if i < iterations - 1:
                time.sleep(interval)
        
        # Calculate summary statistics
        successful_iterations = sum(1 for r in results["results"] if r.get("success", False))
        latencies = [r.get("latency") for r in results["results"] if r.get("success", False) and "latency" in r]
        
        results["summary"] = {
            "successful_iterations": successful_iterations,
            "failed_iterations": iterations - successful_iterations,
            "success_rate": successful_iterations / iterations if iterations > 0 else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else None,
            "min_latency": min(latencies) if latencies else None,
            "max_latency": max(latencies) if latencies else None
        }
        
        return results
    
    def _test_concurrency(self, api_instance, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test API concurrency handling by sending multiple simultaneous requests.
        
        Args:
            api_instance: The API instance to test
            parameters: Test parameters
            
        Returns:
            Test results with concurrency metrics
        """
        # Extract parameters
        prompt = parameters.get("prompt", "Hello, world!")
        concurrency_levels = parameters.get("concurrency_levels", [1, 2, 5, 10])
        requests_per_level = parameters.get("requests_per_level", 5)
        model = parameters.get("model")
        
        results = {
            "status": "success",
            "test_type": "concurrency",
            "api_type": api_instance.__class__.__name__,
            "concurrency_levels": concurrency_levels,
            "requests_per_level": requests_per_level,
            "results_by_level": {},
            "summary": {}
        }
        
        # Define a function to make API requests
        def make_request():
            try:
                start_time = time.time()
                
                # Make API call based on API type
                if hasattr(api_instance, "completion"):
                    response = api_instance.completion(prompt, model=model)
                elif hasattr(api_instance, "chat_completion"):
                    response = api_instance.chat_completion([{"role": "user", "content": prompt}], model=model)
                else:
                    # Use the simplest API call we can find
                    response = api_instance.__test__()
                
                latency = time.time() - start_time
                
                return {
                    "success": True,
                    "latency": latency,
                    "timestamp": start_time
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        # Test each concurrency level
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            level_results = []
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit all requests at once
                futures = [executor.submit(make_request) for _ in range(requests_per_level)]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    level_results.append(result)
            
            # Calculate statistics for this concurrency level
            successful_requests = sum(1 for r in level_results if r.get("success", False))
            latencies = [r.get("latency") for r in level_results if r.get("success", False) and "latency" in r]
            total_time = time.time() - start_time
            
            level_summary = {
                "total_requests": len(level_results),
                "successful_requests": successful_requests,
                "failed_requests": len(level_results) - successful_requests,
                "success_rate": successful_requests / len(level_results) if level_results else 0,
                "avg_latency": sum(latencies) / len(latencies) if latencies else None,
                "min_latency": min(latencies) if latencies else None,
                "max_latency": max(latencies) if latencies else None,
                "total_time": total_time,
                "requests_per_second": successful_requests / total_time if total_time > 0 and successful_requests > 0 else 0
            }
            
            results["results_by_level"][concurrency] = {
                "requests": level_results,
                "summary": level_summary
            }
        
        # Calculate overall summary
        results["summary"] = {
            "optimal_concurrency": None,
            "max_throughput": 0,
            "concurrency_effects": {}
        }
        
        # Find optimal concurrency level based on throughput
        for concurrency, level_data in results["results_by_level"].items():
            throughput = level_data["summary"]["requests_per_second"]
            results["summary"]["concurrency_effects"][concurrency] = {
                "throughput": throughput,
                "success_rate": level_data["summary"]["success_rate"],
                "avg_latency": level_data["summary"]["avg_latency"]
            }
            
            if throughput > results["summary"]["max_throughput"]:
                results["summary"]["max_throughput"] = throughput
                results["summary"]["optimal_concurrency"] = concurrency
        
        return results
    
    def _record_test_result(self, api_type: str, test_type: str, parameters: Dict[str, Any], result: Dict[str, Any]):
        """
        Record test results for metrics and visualization.
        
        Args:
            api_type: The type of API tested
            test_type: The type of test performed
            parameters: Test parameters
            result: Test results
        """
        # Create a record of this test
        test_record = {
            "api_type": api_type,
            "test_type": test_type,
            "parameters": parameters,
            "timestamp": time.time(),
            "summary": result.get("summary", {})
        }
        
        # Add to test history
        self.metrics["test_history"].append(test_record)
        
        # Limit history size
        if len(self.metrics["test_history"]) > 1000:
            self.metrics["test_history"] = self.metrics["test_history"][-1000:]
        
        # Update specific metrics based on test type
        if test_type == "latency" and "summary" in result:
            if api_type not in self.metrics["latency"]:
                self.metrics["latency"][api_type] = []
            
            self.metrics["latency"][api_type].append({
                "timestamp": time.time(),
                "avg_latency": result["summary"].get("avg_latency"),
                "min_latency": result["summary"].get("min_latency"),
                "max_latency": result["summary"].get("max_latency")
            })
        
        elif test_type == "throughput" and "summary" in result:
            if api_type not in self.metrics["throughput"]:
                self.metrics["throughput"][api_type] = []
            
            self.metrics["throughput"][api_type].append({
                "timestamp": time.time(),
                "requests_per_second": result["summary"].get("requests_per_second"),
                "concurrency": parameters.get("concurrency")
            })
        
        elif test_type == "reliability" and "summary" in result:
            if api_type not in self.metrics["reliability"]:
                self.metrics["reliability"][api_type] = []
            
            self.metrics["reliability"][api_type].append({
                "timestamp": time.time(),
                "success_rate": result["summary"].get("success_rate")
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for API testing.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def get_api_performance_history(self, api_type: str, test_type: str = None) -> Dict[str, Any]:
        """
        Get performance history for a specific API type.
        
        Args:
            api_type: The type of API to get metrics for
            test_type: Optional test type to filter by
            
        Returns:
            Dictionary of performance metrics
        """
        # Filter test history by API type and optionally by test type
        if test_type:
            history = [
                record for record in self.metrics["test_history"]
                if record["api_type"] == api_type and record["test_type"] == test_type
            ]
        else:
            history = [
                record for record in self.metrics["test_history"]
                if record["api_type"] == api_type
            ]
        
        # Get specific metrics based on available test types
        metrics = {}
        
        # Extract latency metrics if available
        if api_type in self.metrics["latency"]:
            metrics["latency"] = self.metrics["latency"][api_type]
        
        # Extract throughput metrics if available
        if api_type in self.metrics["throughput"]:
            metrics["throughput"] = self.metrics["throughput"][api_type]
        
        # Extract reliability metrics if available
        if api_type in self.metrics["reliability"]:
            metrics["reliability"] = self.metrics["reliability"][api_type]
        
        return {
            "api_type": api_type,
            "test_type": test_type,
            "history": history,
            "metrics": metrics
        }
    
    def compare_apis(self, api_types: List[str], test_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple API backends on the same test.
        
        Args:
            api_types: List of API types to compare
            test_type: The type of test to run
            parameters: Test parameters
            
        Returns:
            Comparison results
        """
        results = {
            "status": "success",
            "test_type": test_type,
            "parameters": parameters,
            "results_by_api": {},
            "summary": {}
        }
        
        # Run the same test for each API type
        for api_type in api_types:
            # Check if API type is available
            if api_type not in self.available_api_types:
                results["results_by_api"][api_type] = {
                    "status": "error",
                    "error": f"API type '{api_type}' not available"
                }
                continue
            
            # Register the API backend if not already registered
            if not self.register_api_backend(api_type):
                results["results_by_api"][api_type] = {
                    "status": "error",
                    "error": f"Failed to register API backend '{api_type}'"
                }
                continue
            
            # Run the test
            test_result = self.run_distributed_test(api_type, test_type, parameters)
            
            # Store the result
            results["results_by_api"][api_type] = test_result
        
        # Create comparison summary based on test type
        if test_type == "latency":
            latency_comparison = {}
            for api_type, result in results["results_by_api"].items():
                if result.get("status") == "success" and "summary" in result:
                    latency_comparison[api_type] = result["summary"].get("avg_latency")
            
            # Sort by average latency
            sorted_apis = sorted(latency_comparison.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
            
            results["summary"]["latency_ranking"] = [api for api, _ in sorted_apis]
            results["summary"]["fastest_api"] = sorted_apis[0][0] if sorted_apis else None
            
        elif test_type == "throughput":
            throughput_comparison = {}
            for api_type, result in results["results_by_api"].items():
                if result.get("status") == "success" and "summary" in result:
                    throughput_comparison[api_type] = result["summary"].get("requests_per_second")
            
            # Sort by throughput (highest first)
            sorted_apis = sorted(throughput_comparison.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
            
            results["summary"]["throughput_ranking"] = [api for api, _ in sorted_apis]
            results["summary"]["highest_throughput_api"] = sorted_apis[0][0] if sorted_apis else None
            
        elif test_type == "reliability":
            reliability_comparison = {}
            for api_type, result in results["results_by_api"].items():
                if result.get("status") == "success" and "summary" in result:
                    reliability_comparison[api_type] = result["summary"].get("success_rate")
            
            # Sort by reliability (highest first)
            sorted_apis = sorted(reliability_comparison.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
            
            results["summary"]["reliability_ranking"] = [api for api, _ in sorted_apis]
            results["summary"]["most_reliable_api"] = sorted_apis[0][0] if sorted_apis else None
        
        return results
    
    def get_api_backend_info(self, api_type: str = None) -> Dict[str, Any]:
        """
        Get information about API backends.
        
        Args:
            api_type: Optional specific API type to get info for
            
        Returns:
            Dictionary of API backend information
        """
        result = {
            "available_api_types": list(self.available_api_types),
            "registered_api_types": list(self.api_instances.keys()),
            "api_backends": {}
        }
        
        # If a specific API type is requested, only include that one
        if api_type:
            if api_type not in self.available_api_types:
                return {
                    "status": "error",
                    "error": f"API type '{api_type}' not available"
                }
            
            api_types = [api_type]
        else:
            api_types = self.available_api_types
        
        # Get information for each API type
        for api in api_types:
            # Check if API is registered
            is_registered = api in self.api_instances
            
            api_info = {
                "registered": is_registered,
                "module": self.api_backends[api].__name__ if api in self.api_backends else None,
                "capabilities": []
            }
            
            # Get capabilities if registered
            if is_registered:
                instance = self.api_instances[api]
                
                # Check for common API methods
                if hasattr(instance, "completion"):
                    api_info["capabilities"].append("completion")
                if hasattr(instance, "chat_completion"):
                    api_info["capabilities"].append("chat_completion")
                if hasattr(instance, "embedding"):
                    api_info["capabilities"].append("embedding")
                if hasattr(instance, "streaming"):
                    api_info["capabilities"].append("streaming")
            
            result["api_backends"][api] = api_info
        
        return result
    
    def detect_anomalies(self, api_type: str, test_type: str, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Detect anomalies in API performance metrics.
        
        Args:
            api_type: The API type to analyze
            test_type: The test type to analyze
            lookback_days: Number of days to look back for anomalies
            
        Returns:
            Dictionary with detected anomalies
        """
        # Get the ML anomaly detection component from the framework
        ml_detector = self.framework.monitoring.ml_detector if hasattr(self.framework.monitoring, "ml_detector") else None
        
        if not ml_detector:
            return {
                "status": "error",
                "error": "ML anomaly detection not available"
            }
        
        # Get metrics for the specified API and test type
        metrics_data = []
        lookback_time = time.time() - (lookback_days * 24 * 60 * 60)
        
        # Get metrics based on test type
        if test_type == "latency" and api_type in self.metrics["latency"]:
            metrics_data = [
                (record["timestamp"], record["avg_latency"]) 
                for record in self.metrics["latency"][api_type] 
                if record["timestamp"] >= lookback_time and record["avg_latency"] is not None
            ]
            metric_name = f"{api_type}_latency"
            
        elif test_type == "throughput" and api_type in self.metrics["throughput"]:
            metrics_data = [
                (record["timestamp"], record["requests_per_second"]) 
                for record in self.metrics["throughput"][api_type] 
                if record["timestamp"] >= lookback_time and record["requests_per_second"] is not None
            ]
            metric_name = f"{api_type}_throughput"
            
        elif test_type == "reliability" and api_type in self.metrics["reliability"]:
            metrics_data = [
                (record["timestamp"], record["success_rate"]) 
                for record in self.metrics["reliability"][api_type] 
                if record["timestamp"] >= lookback_time and record["success_rate"] is not None
            ]
            metric_name = f"{api_type}_reliability"
            
        else:
            return {
                "status": "error",
                "error": f"No metrics data available for API '{api_type}' and test type '{test_type}'"
            }
        
        # Check if we have enough data for analysis
        if len(metrics_data) < 10:
            return {
                "status": "warning",
                "warning": f"Not enough data for anomaly detection (need at least 10 data points, got {len(metrics_data)})"
            }
        
        # Detect anomalies
        anomalies = ml_detector.detect_anomalies(metrics_data, metric_name=metric_name)
        
        # Generate forecast
        forecast = ml_detector.forecast_trend(metrics_data, metric_name=metric_name, forecast_periods=24)
        
        return {
            "status": "success",
            "api_type": api_type,
            "test_type": test_type,
            "lookback_days": lookback_days,
            "data_points": len(metrics_data),
            "anomalies": anomalies,
            "forecast": forecast
        }
    
    def get_api_framework_status(self) -> Dict[str, Any]:
        """
        Get status of the API testing framework.
        
        Returns:
            Dictionary with framework status
        """
        # Get health check from the framework
        framework_health = self.framework.health_check()
        
        # Get API backend status
        api_status = {
            api_type: {
                "registered": True,
                "instance": api_instance.__class__.__name__
            }
            for api_type, api_instance in self.api_instances.items()
        }
        
        # Get metrics summary
        metrics_summary = {
            "test_history_count": len(self.metrics["test_history"]),
            "apis_with_latency_data": list(self.metrics["latency"].keys()),
            "apis_with_throughput_data": list(self.metrics["throughput"].keys()),
            "apis_with_reliability_data": list(self.metrics["reliability"].keys())
        }
        
        return {
            "status": "running",
            "framework_health": framework_health,
            "api_status": api_status,
            "available_api_types": list(self.available_api_types),
            "registered_api_types": list(self.api_instances.keys()),
            "metrics_summary": metrics_summary,
            "uptime": time.time() - self.framework.start_time if hasattr(self.framework, "start_time") else 0
        }
    
    def stop(self):
        """Stop the API testing framework."""
        self.framework.stop()
        logger.info("API distributed testing framework stopped")


def run_example_test():
    """Run an example test using the APIDistributedTesting framework."""
    # Initialize the framework
    api_testing = APIDistributedTesting()
    
    # Get available API types
    api_info = api_testing.get_api_backend_info()
    print(f"Available API types: {api_info['available_api_types']}")
    
    # Register an API backend (if available)
    available_apis = api_info['available_api_types']
    if not available_apis:
        print("No API backends available. Please check your environment.")
        return
    
    # Choose the first available API
    api_type = available_apis[0]
    print(f"Testing with API type: {api_type}")
    
    # Register the API
    success = api_testing.register_api_backend(api_type)
    if not success:
        print(f"Failed to register API backend: {api_type}")
        return
    
    # Run a latency test
    print(f"Running latency test for {api_type}...")
    latency_result = api_testing.run_distributed_test(
        api_type=api_type,
        test_type=APITestType.LATENCY,
        parameters={
            "prompt": "What is the capital of France?",
            "iterations": 3
        }
    )
    
    # Print the results
    print(f"Latency test results for {api_type}:")
    print(f"  Status: {latency_result['status']}")
    if "summary" in latency_result:
        print(f"  Average latency: {latency_result['summary'].get('avg_latency', 'N/A'):.3f}s")
        print(f"  Min latency: {latency_result['summary'].get('min_latency', 'N/A'):.3f}s")
        print(f"  Max latency: {latency_result['summary'].get('max_latency', 'N/A'):.3f}s")
        print(f"  Successful iterations: {latency_result['summary'].get('successful_iterations', 0)}")
    
    # Get framework status
    framework_status = api_testing.get_api_framework_status()
    print("\nAPI Framework Status:")
    print(f"  Status: {framework_status['status']}")
    print(f"  Available API types: {framework_status['available_api_types']}")
    print(f"  Registered API types: {framework_status['registered_api_types']}")
    print(f"  Framework health: {framework_status['framework_health']['status']}")
    
    # Stop the framework
    api_testing.stop()
    print("API testing framework stopped")


if __name__ == "__main__":
    run_example_test()