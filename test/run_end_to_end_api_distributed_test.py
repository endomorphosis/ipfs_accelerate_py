#!/usr/bin/env python3
"""
End-to-End API Distributed Testing Example

This script demonstrates a complete end-to-end workflow for the API Distributed Testing Framework.
It creates a simulated environment with coordinator and workers for testing API providers.

Features:
- Simulated coordinator and worker nodes
- Multiple API providers (OpenAI, Claude, Groq)
- Different test types (latency, throughput, reliability, cost_efficiency)
- Performance metrics collection and visualization
- Anomaly detection and predictive analytics
- Multiple execution modes (simulation, multiprocess, distributed)
"""

import os
import sys
import time
import json
import uuid
import signal
import logging
import argparse
import datetime
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import random
import platform
import traceback
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_COORDINATOR_PORT = 5555
DEFAULT_NUM_WORKERS = 3
DEFAULT_OUTPUT_DIR = "./api_test_results"
DEFAULT_TEST_SUITE = "basic"
DEFAULT_MODE = "simulation"

# Define basic task class for simulation
class Task:
    """Simple task class for simulation."""
    
    def __init__(self, task_id: str, task_type: str, parameters: Dict[str, Any], priority: int = 0, tags: List[str] = None):
        """
        Initialize a task.
        
        Args:
            task_id: Task ID
            task_type: Task type
            parameters: Task parameters
            priority: Task priority
            tags: Task tags
        """
        self.task_id = task_id
        self.task_type = task_type
        self.parameters = parameters
        self.priority = priority
        self.tags = tags or []


class SimulatedAPIProvider:
    """Simulated API provider for testing without actual API calls."""
    
    def __init__(self, provider_name: str, model_name: str = "default-model"):
        """
        Initialize a simulated API provider.
        
        Args:
            provider_name: Name of the provider
            model_name: Name of the model
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.latency_base = self._get_base_latency()
        self.throughput_base = self._get_base_throughput()
        self.reliability_base = self._get_base_reliability()
        self.tokens_per_dollar = self._get_tokens_per_dollar()
        
    def _get_base_latency(self) -> float:
        """Get base latency in ms for this provider."""
        # Simulate different latency characteristics for different providers
        if self.provider_name.lower() == "openai":
            return 250.0  # 250ms base latency
        elif self.provider_name.lower() == "claude":
            return 300.0  # 300ms base latency
        elif self.provider_name.lower() == "groq":
            return 80.0   # 80ms base latency
        else:
            return 200.0  # Default case
    
    def _get_base_throughput(self) -> float:
        """Get base throughput in requests per second."""
        # Simulate different throughput characteristics for different providers
        if self.provider_name.lower() == "openai":
            return 10.0   # 10 requests per second
        elif self.provider_name.lower() == "claude":
            return 8.0    # 8 requests per second
        elif self.provider_name.lower() == "groq":
            return 15.0   # 15 requests per second
        else:
            return 5.0    # Default case
    
    def _get_base_reliability(self) -> float:
        """Get base success rate (0.0-1.0)."""
        # Simulate different reliability characteristics for different providers
        if self.provider_name.lower() == "openai":
            return 0.995  # 99.5% success rate
        elif self.provider_name.lower() == "claude":
            return 0.990  # 99.0% success rate
        elif self.provider_name.lower() == "groq":
            return 0.980  # 98.0% success rate
        else:
            return 0.950  # Default case
    
    def _get_tokens_per_dollar(self) -> float:
        """Get tokens per dollar for cost efficiency."""
        # Simulate different cost efficiency for different providers
        if self.provider_name.lower() == "openai":
            return 80000.0  # 80K tokens per dollar
        elif self.provider_name.lower() == "claude":
            return 60000.0  # 60K tokens per dollar
        elif self.provider_name.lower() == "groq":
            return 120000.0 # 120K tokens per dollar
        else:
            return 50000.0  # Default case
    
    def simulate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Simulate a chat completion request.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Simulated response
        """
        # Simulate processing time
        prompt_length = sum(len(msg.get("content", "")) for msg in messages)
        
        # Add some randomness to latency
        latency_factor = random.uniform(0.8, 1.2)
        processing_time = self.latency_base * latency_factor * (1 + (prompt_length / 1000))
        
        # Simulate potential errors based on reliability
        if random.random() > self.reliability_base:
            # Simulate an error
            error_types = ["rate_limit", "timeout", "internal_error", "bad_request"]
            error_type = random.choice(error_types)
            
            if error_type == "rate_limit":
                raise Exception("API rate limit exceeded")
            elif error_type == "timeout":
                raise Exception("Request timed out")
            elif error_type == "internal_error":
                raise Exception("Internal server error")
            else:
                raise Exception("Bad request")
        
        # Simulate token counts
        input_tokens = prompt_length // 4  # Rough approximation of tokens
        output_tokens = random.randint(50, 200)  # Random response length
        
        # Add some delay to simulate actual API call
        delay = processing_time / 1000.0  # Convert ms to seconds
        time.sleep(min(delay, 0.1))  # Cap at 100ms to keep tests fast
        
        # Create simulated response
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Simulated response from {self.provider_name}"
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "simulated": True,
            "latency_ms": processing_time
        }


class SimulatedCoordinator:
    """Simulated coordinator for end-to-end testing."""
    
    def __init__(self):
        """Initialize the simulated coordinator."""
        self.tasks = {}
        self.results = {}
        self.workers = {}
        self.lock = threading.Lock()
    
    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> str:
        """Register a worker with the coordinator."""
        with self.lock:
            self.workers[worker_id] = {
                "id": worker_id,
                "capabilities": capabilities,
                "last_heartbeat": time.time(),
                "status": "active"
            }
        return worker_id
    
    def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker from the coordinator."""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                return True
            return False
    
    def create_task(self, api_type: str, test_type: str, parameters: Dict[str, Any]) -> str:
        """
        Create a task for a worker to process.
        
        Args:
            api_type: API provider type
            test_type: Test type
            parameters: Test parameters
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "api_type": api_type,
            "test_type": test_type,
            "parameters": parameters,
            "status": "pending",
            "created_at": time.time()
        }
        
        with self.lock:
            self.tasks[task_id] = task
        
        return task_id
    
    def get_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the next task for a worker to process.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Task data or None if no task is available
        """
        with self.lock:
            # Check if worker is registered
            if worker_id not in self.workers:
                return None
            
            # Get worker capabilities
            capabilities = self.workers[worker_id]["capabilities"]
            supported_apis = capabilities.get("providers", [])
            
            # Find a suitable task
            for task_id, task in self.tasks.items():
                if task["status"] == "pending":
                    # Check if this worker can handle this API type
                    api_type = task["api_type"].lower()
                    if api_type in [p.lower() for p in supported_apis]:
                        # Assign task to worker
                        task["status"] = "running"
                        task["worker_id"] = worker_id
                        task["started_at"] = time.time()
                        
                        # Convert to Task object
                        return Task(
                            task_id=task_id,
                            task_type="api_test",
                            parameters={
                                "api_type": task["api_type"],
                                "test_type": task["test_type"],
                                "test_parameters": task["parameters"]
                            },
                            priority=0,
                            tags=[]
                        )
            
            return None
    
    def submit_result(self, worker_id: str, task_id: str, result: Dict[str, Any]) -> None:
        """
        Submit a task result.
        
        Args:
            worker_id: Worker ID
            task_id: Task ID
            result: Task result data
        """
        with self.lock:
            # Check if task exists
            if task_id not in self.tasks:
                return
            
            # Update task status
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["completed_at"] = time.time()
            
            # Store result
            self.results[task_id] = {
                "task_id": task_id,
                "worker_id": worker_id,
                "result": result,
                "created_at": self.tasks[task_id]["created_at"],
                "completed_at": time.time()
            }
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task result.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result data or None if not found
        """
        with self.lock:
            return self.results.get(task_id)
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all task results.
        
        Returns:
            Dictionary mapping task IDs to result data
        """
        with self.lock:
            return self.results.copy()


class SimulatedWorker:
    """Simulated worker for end-to-end testing."""
    
    def __init__(self, worker_id: str, coordinator: SimulatedCoordinator, supported_apis: List[str]):
        """
        Initialize the simulated worker.
        
        Args:
            worker_id: Worker ID
            coordinator: Simulated coordinator
            supported_apis: List of supported API providers
        """
        self.worker_id = worker_id
        self.coordinator = coordinator
        self.supported_apis = supported_apis
        self.running = False
        self.thread = None
        
        # Register with coordinator
        self.coordinator.register_worker(
            worker_id=self.worker_id,
            capabilities={
                "providers": supported_apis,
                "system": {
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "processor": platform.processor()
                }
            }
        )
    
    def start(self) -> None:
        """Start the worker."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        # Unregister from coordinator
        self.coordinator.unregister_worker(self.worker_id)
    
    def _worker_loop(self) -> None:
        """Worker main loop."""
        while self.running:
            # Get a task from the coordinator
            task = self.coordinator.get_task(self.worker_id)
            
            if task:
                logger.info(f"Worker {self.worker_id} processing task {task.task_id}")
                
                # Process the task
                result = self._process_task(task)
                
                # Submit the result
                self.coordinator.submit_result(
                    worker_id=self.worker_id,
                    task_id=task.task_id,
                    result=result
                )
                
                logger.info(f"Worker {self.worker_id} completed task {task.task_id}")
            else:
                # No task available, sleep for a bit
                time.sleep(0.1)
    
    def _process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result data
        """
        try:
            # Extract parameters
            api_type = task.parameters.get("api_type")
            test_type = task.parameters.get("test_type")
            test_parameters = task.parameters.get("test_parameters", {})
            
            # Create simulated API provider
            provider = SimulatedAPIProvider(api_type)
            
            # Gather test results based on test type
            if test_type == "latency":
                return self._run_latency_test(provider, test_parameters)
            elif test_type == "throughput":
                return self._run_throughput_test(provider, test_parameters)
            elif test_type == "reliability":
                return self._run_reliability_test(provider, test_parameters)
            elif test_type == "cost_efficiency":
                return self._run_cost_efficiency_test(provider, test_parameters)
            else:
                return {
                    "error": f"Unsupported test type: {test_type}"
                }
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            return {
                "error": str(e)
            }
    
    def _run_latency_test(self, provider: SimulatedAPIProvider, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a latency test.
        
        Args:
            provider: Simulated API provider
            parameters: Test parameters
            
        Returns:
            Test result data
        """
        messages = parameters.get("messages", [{"role": "user", "content": "Hello, world!"}])
        iterations = parameters.get("iterations", 5)
        
        latencies = []
        errors = []
        
        for i in range(iterations):
            try:
                # Simulate chat completion
                start_time = time.time()
                response = provider.simulate_chat_completion(messages)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                errors.append(str(e))
        
        # Calculate statistics
        import numpy as np
        
        results = {
            "test_type": "latency",
            "iterations": iterations,
            "successful_iterations": len(latencies),
            "failed_iterations": len(errors),
            "errors": errors,
            "latencies_ms": latencies
        }
        
        if latencies:
            results.update({
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "stddev_latency_ms": float(np.std(latencies)),
                "percentiles_ms": {
                    "50": float(np.percentile(latencies, 50)),
                    "90": float(np.percentile(latencies, 90)),
                    "95": float(np.percentile(latencies, 95)),
                    "99": float(np.percentile(latencies, 99))
                }
            })
        
        return results
    
    def _run_throughput_test(self, provider: SimulatedAPIProvider, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a throughput test.
        
        Args:
            provider: Simulated API provider
            parameters: Test parameters
            
        Returns:
            Test result data
        """
        messages = parameters.get("messages", [{"role": "user", "content": "Hello, world!"}])
        duration = parameters.get("duration", 5)
        concurrent_requests = parameters.get("concurrent_requests", 3)
        
        # Use local variables for thread-safe counters
        requests_completed = 0
        errors = []
        latencies = []
        
        # Create a lock for thread-safe incrementing
        lock = threading.Lock()
        
        # Flag to signal threads to stop
        stop_flag = threading.Event()
        
        def worker():
            nonlocal requests_completed, errors, latencies
            
            while not stop_flag.is_set():
                try:
                    # Simulate chat completion
                    start_time = time.time()
                    response = provider.simulate_chat_completion(messages)
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    
                    with lock:
                        requests_completed += 1
                        latencies.append(latency)
                except Exception as e:
                    with lock:
                        errors.append(str(e))
        
        # Start worker threads
        threads = []
        for _ in range(concurrent_requests):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Run for specified duration
        start_time = time.time()
        time.sleep(duration)
        stop_flag.set()
        
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=1)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate statistics
        import numpy as np
        
        results = {
            "test_type": "throughput",
            "duration_seconds": actual_duration,
            "concurrent_requests": concurrent_requests,
            "requests_completed": requests_completed,
            "errors": len(errors),
            "error_messages": errors[:10],  # Limit to first 10 errors
            "throughput_rps": requests_completed / actual_duration
        }
        
        if latencies:
            results.update({
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "stddev_latency_ms": float(np.std(latencies)),
                "latency_percentiles_ms": {
                    "50": float(np.percentile(latencies, 50)),
                    "90": float(np.percentile(latencies, 90)),
                    "95": float(np.percentile(latencies, 95)),
                    "99": float(np.percentile(latencies, 99))
                }
            })
        
        return results
    
    def _run_reliability_test(self, provider: SimulatedAPIProvider, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a reliability test.
        
        Args:
            provider: Simulated API provider
            parameters: Test parameters
            
        Returns:
            Test result data
        """
        messages = parameters.get("messages", [{"role": "user", "content": "Hello, world!"}])
        iterations = parameters.get("iterations", 50)
        
        successes = 0
        failures = 0
        errors = []
        
        for _ in range(iterations):
            try:
                # Simulate chat completion
                response = provider.simulate_chat_completion(messages)
                successes += 1
            except Exception as e:
                failures += 1
                errors.append(str(e))
        
        # Calculate statistics
        results = {
            "test_type": "reliability",
            "iterations": iterations,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / iterations if iterations > 0 else 0,
            "error_rate": failures / iterations if iterations > 0 else 0,
            "errors": errors[:10]  # Limit to first 10 errors
        }
        
        return results
    
    def _run_cost_efficiency_test(self, provider: SimulatedAPIProvider, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a cost efficiency test.
        
        Args:
            provider: Simulated API provider
            parameters: Test parameters
            
        Returns:
            Test result data
        """
        messages = parameters.get("messages", [{"role": "user", "content": "Hello, world!"}])
        iterations = parameters.get("iterations", 5)
        
        total_tokens = 0
        total_cost = 0
        latencies = []
        
        # Approximate cost per 1K tokens
        cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens (simulated)
        
        for _ in range(iterations):
            try:
                # Simulate chat completion
                start_time = time.time()
                response = provider.simulate_chat_completion(messages)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                # Extract token counts
                usage = response.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                
                total_tokens += tokens
                
                # Calculate cost
                cost = (tokens / 1000) * cost_per_1k_tokens
                total_cost += cost
            except Exception:
                pass
        
        # Calculate statistics
        import numpy as np
        
        # Only calculate if we have successful iterations
        if latencies:
            avg_tokens = total_tokens / len(latencies)
            avg_cost = total_cost / len(latencies)
            avg_latency = float(np.mean(latencies))
            
            # Cost efficiency metrics
            tokens_per_dollar = provider.tokens_per_dollar
            cost_per_second = avg_cost / (avg_latency / 1000)
            tokens_per_second = avg_tokens / (avg_latency / 1000)
            
            results = {
                "test_type": "cost_efficiency",
                "iterations": iterations,
                "successful_iterations": len(latencies),
                "avg_tokens": avg_tokens,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "avg_cost": avg_cost,
                "avg_latency_ms": avg_latency,
                "cost_efficiency_metrics": {
                    "tokens_per_dollar": tokens_per_dollar,
                    "cost_per_second": cost_per_second,
                    "tokens_per_second": tokens_per_second
                }
            }
        else:
            results = {
                "test_type": "cost_efficiency",
                "iterations": iterations,
                "successful_iterations": 0,
                "error": "No successful iterations"
            }
        
        return results


class EndToEndAPITest:
    """End-to-end test orchestrator for the API Distributed Testing Framework."""
    
    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        coordinator_host: str = "localhost",
        coordinator_port: int = DEFAULT_COORDINATOR_PORT,
        num_workers: int = DEFAULT_NUM_WORKERS,
        apis: List[str] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        test_suite: str = DEFAULT_TEST_SUITE
    ):
        """
        Initialize the end-to-end test orchestrator.
        
        Args:
            mode: Test mode (simulation, multiprocess, distributed)
            coordinator_host: Coordinator host
            coordinator_port: Coordinator port
            num_workers: Number of worker nodes
            apis: List of API providers to test
            output_dir: Output directory for test results
            test_suite: Test suite to run
        """
        self.mode = mode
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.num_workers = num_workers
        self.apis = apis or ["openai", "claude", "groq"]
        self.output_dir = output_dir
        self.test_suite = test_suite
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components based on mode
        if mode == "simulation":
            self.coordinator = SimulatedCoordinator()
            self.workers = []
            
            # Create simulated workers
            for i in range(num_workers):
                # Assign different API providers to different workers
                if i == 0:
                    supported_apis = ["openai", "groq"]
                elif i == 1:
                    supported_apis = ["claude", "groq"]
                else:
                    supported_apis = self.apis
                
                worker = SimulatedWorker(
                    worker_id=f"worker-{i+1}",
                    coordinator=self.coordinator,
                    supported_apis=supported_apis
                )
                self.workers.append(worker)
        
        # Initialize test parameters
        self.test_parameters = self._get_test_parameters()
    
    def _get_test_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get test parameters for different test suites.
        
        Returns:
            Dictionary mapping test suite names to parameters
        """
        messages = [{"role": "user", "content": "Explain quantum computing in simple terms."}]
        
        return {
            "basic": {
                "latency": {
                    "messages": messages,
                    "iterations": 5
                },
                "throughput": {
                    "messages": messages,
                    "duration": 3,
                    "concurrent_requests": 3
                },
                "reliability": {
                    "messages": messages,
                    "iterations": 20
                },
                "cost_efficiency": {
                    "messages": messages,
                    "iterations": 3
                }
            },
            "comprehensive": {
                "latency": {
                    "messages": messages,
                    "iterations": 10
                },
                "throughput": {
                    "messages": messages,
                    "duration": 5,
                    "concurrent_requests": 5
                },
                "reliability": {
                    "messages": messages,
                    "iterations": 50
                },
                "cost_efficiency": {
                    "messages": messages,
                    "iterations": 5
                }
            },
            "stress": {
                "latency": {
                    "messages": messages,
                    "iterations": 20
                },
                "throughput": {
                    "messages": messages,
                    "duration": 10,
                    "concurrent_requests": 10
                },
                "reliability": {
                    "messages": messages,
                    "iterations": 100
                },
                "cost_efficiency": {
                    "messages": messages,
                    "iterations": 10
                }
            }
        }
    
    def start_infrastructure(self):
        """Start the testing infrastructure (coordinator and workers)."""
        if self.mode == "simulation":
            # Start workers
            for worker in self.workers:
                worker.start()
        
        logger.info(f"Started testing infrastructure in {self.mode} mode")
    
    def stop_infrastructure(self):
        """Stop the testing infrastructure."""
        if self.mode == "simulation":
            # Stop workers
            for worker in self.workers:
                worker.stop()
        
        logger.info(f"Stopped testing infrastructure in {self.mode} mode")
    
    def run_test_suite(self) -> Dict[str, Any]:
        """
        Run the selected test suite and return results.
        
        Returns:
            Test results dictionary
        """
        # Use parameters based on test suite
        if self.test_suite not in self.test_parameters:
            logger.warning(f"Unknown test suite: {self.test_suite}, falling back to basic")
            self.test_suite = "basic"
        
        parameters = self.test_parameters[self.test_suite]
        
        # Start infrastructure
        self.start_infrastructure()
        
        try:
            # Create tasks for each API and test type
            tasks = []
            
            test_types = ["latency", "throughput", "reliability", "cost_efficiency"]
            
            # Create tasks for each API provider and test type
            for api in self.apis:
                for test_type in test_types:
                    test_params = parameters.get(test_type, {})
                    
                    # Create task
                    if self.mode == "simulation":
                        task_id = self.coordinator.create_task(
                            api_type=api,
                            test_type=test_type,
                            parameters=test_params
                        )
                        tasks.append((api, test_type, task_id))
            
            # Wait for tasks to complete
            results = {}
            
            for api, test_type, task_id in tasks:
                # Wait for result with timeout
                start_time = time.time()
                result = None
                
                while time.time() - start_time < 60:  # 60 second timeout
                    if self.mode == "simulation":
                        result = self.coordinator.get_result(task_id)
                    
                    if result:
                        break
                    
                    time.sleep(0.1)
                
                # Store result
                if result:
                    if api not in results:
                        results[api] = {}
                    
                    results[api][test_type] = result.get("result", {})
                else:
                    logger.warning(f"No result received for {api} {test_type} test")
            
            # Analyze results
            self._analyze_results(results)
            
            return results
        finally:
            # Stop infrastructure
            self.stop_infrastructure()
    
    def _analyze_results(self, results: Dict[str, Any]):
        """
        Analyze test results and add summary metrics.
        
        Args:
            results: Raw test results
        """
        # Add summary metrics for each API provider
        for api, api_results in results.items():
            summary = {
                "provider": api,
                "tests_completed": len(api_results),
                "latency_ms": None,
                "throughput_rps": None,
                "reliability": None,
                "cost_efficiency": None
            }
            
            # Extract key metrics
            if "latency" in api_results:
                latency_result = api_results["latency"]
                summary["latency_ms"] = latency_result.get("mean_latency_ms")
            
            if "throughput" in api_results:
                throughput_result = api_results["throughput"]
                summary["throughput_rps"] = throughput_result.get("throughput_rps")
            
            if "reliability" in api_results:
                reliability_result = api_results["reliability"]
                summary["reliability"] = reliability_result.get("success_rate")
            
            if "cost_efficiency" in api_results:
                cost_result = api_results["cost_efficiency"]
                if "cost_efficiency_metrics" in cost_result:
                    summary["cost_efficiency"] = cost_result["cost_efficiency_metrics"].get("tokens_per_dollar")
            
            # Add summary to results
            api_results["summary"] = summary
        
        # Create performance ranking
        rankings = {
            "latency": self._rank_apis_by_metric(results, "latency_ms", lower_is_better=True),
            "throughput": self._rank_apis_by_metric(results, "throughput_rps", lower_is_better=False),
            "reliability": self._rank_apis_by_metric(results, "reliability", lower_is_better=False),
            "cost_efficiency": self._rank_apis_by_metric(results, "cost_efficiency", lower_is_better=False)
        }
        
        # Add rankings to results
        results["rankings"] = rankings
        
        # Add overall winner
        results["overall_winner"] = self._determine_overall_winner(results)
        
        # Save complete results to file
        timestamp = int(time.time())
        result_file = os.path.join(self.output_dir, f"end_to_end_test_{timestamp}.json")
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {result_file}")
    
    def _rank_apis_by_metric(self, results: Dict[str, Any], metric: str, lower_is_better: bool) -> List[Dict[str, Any]]:
        """
        Rank API providers by a specific metric.
        
        Args:
            results: Test results
            metric: Metric to rank by
            lower_is_better: Whether lower values are better
            
        Returns:
            List of ranked API providers with scores
        """
        ranking = []
        
        for api, api_results in results.items():
            if "summary" in api_results and api_results["summary"].get(metric) is not None:
                ranking.append({
                    "provider": api,
                    "score": api_results["summary"][metric]
                })
        
        # Sort by score
        ranking.sort(key=lambda x: x["score"], reverse=not lower_is_better)
        
        # Add rank
        for i, item in enumerate(ranking):
            item["rank"] = i + 1
        
        return ranking
    
    def _determine_overall_winner(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the overall best performing API provider.
        
        Args:
            results: Test results
            
        Returns:
            Overall winner information
        """
        # Calculate total ranks
        total_ranks = defaultdict(int)
        
        for metric, ranking in results.get("rankings", {}).items():
            for item in ranking:
                total_ranks[item["provider"]] += item["rank"]
        
        # Find provider with lowest total rank (best overall)
        if total_ranks:
            winner = min(total_ranks.items(), key=lambda x: x[1])
            
            return {
                "provider": winner[0],
                "total_rank": winner[1],
                "results": {
                    metric: next((item for item in ranking if item["provider"] == winner[0]), None)
                    for metric, ranking in results.get("rankings", {}).items()
                }
            }
        else:
            return {"provider": "unknown", "total_rank": 0}


def run_simulation():
    """Run an end-to-end simulation."""
    # Create test orchestrator
    test = EndToEndAPITest(
        mode="simulation",
        num_workers=3,
        apis=["openai", "claude", "groq"],
        test_suite="basic"
    )
    
    # Run test suite
    logger.info("Starting end-to-end API test simulation")
    results = test.run_test_suite()
    
    # Print summary
    print("\nTest Results Summary:")
    print("=====================")
    
    for api, api_results in results.items():
        if api in ["rankings", "overall_winner"]:
            continue
        
        if "summary" in api_results:
            summary = api_results["summary"]
            print(f"\n{summary['provider'].upper()}:")
            
            if summary.get("latency_ms") is not None:
                print(f"  Latency: {summary['latency_ms']:.2f} ms")
            
            if summary.get("throughput_rps") is not None:
                print(f"  Throughput: {summary['throughput_rps']:.2f} req/sec")
            
            if summary.get("reliability") is not None:
                print(f"  Reliability: {summary['reliability']:.1%}")
            
            if summary.get("cost_efficiency") is not None:
                print(f"  Cost efficiency: {summary['cost_efficiency']:.0f} tokens/$")
    
    # Print rankings
    print("\nRankings:")
    print("=========")
    
    for metric, ranking in results.get("rankings", {}).items():
        print(f"\n{metric.capitalize()}:")
        for item in ranking:
            print(f"  #{item['rank']}: {item['provider']} - {item['score']:.2f}")
    
    # Print overall winner
    if "overall_winner" in results:
        winner = results["overall_winner"]
        print(f"\nOverall Winner: {winner['provider'].upper()}")
    
    logger.info("End-to-end API test simulation completed")
    
    return results


def main():
    """Command-line interface for the End-to-End API Distributed Testing Example."""
    parser = argparse.ArgumentParser(
        description="End-to-End API Distributed Testing Example"
    )
    
    # Test mode
    parser.add_argument(
        "--mode",
        choices=["simulation", "multiprocess", "distributed"],
        default=DEFAULT_MODE,
        help="Test execution mode"
    )
    
    # Test suite
    parser.add_argument(
        "--test-suite",
        choices=["basic", "comprehensive", "stress"],
        default=DEFAULT_TEST_SUITE,
        help="Test suite to run"
    )
    
    # APIs to test
    parser.add_argument(
        "--apis",
        type=str,
        default="openai,claude,groq",
        help="Comma-separated list of API providers to test"
    )
    
    # Coordinator configuration (for distributed mode)
    parser.add_argument(
        "--coordinator-host",
        type=str,
        default="localhost",
        help="Coordinator host (for distributed mode)"
    )
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=DEFAULT_COORDINATOR_PORT,
        help="Coordinator port (for distributed mode)"
    )
    
    # Worker configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker nodes"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for test results"
    )
    
    args = parser.parse_args()
    
    # Parse API list
    apis = [api.strip() for api in args.apis.split(",")]
    
    if args.mode == "simulation":
        # Run in simulation mode
        test = EndToEndAPITest(
            mode=args.mode,
            coordinator_host=args.coordinator_host,
            coordinator_port=args.coordinator_port,
            num_workers=args.num_workers,
            apis=apis,
            output_dir=args.output_dir,
            test_suite=args.test_suite
        )
        
        # Register signal handler for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Interrupt received, stopping test...")
            test.stop_infrastructure()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Run test suite
        results = test.run_test_suite()
        
        # Print overall winner
        if "overall_winner" in results:
            winner = results["overall_winner"]
            print(f"\nOverall Winner: {winner['provider'].upper()}")
        
        return 0
    elif args.mode == "multiprocess":
        logger.error("Multiprocess mode not implemented yet")
        return 1
    elif args.mode == "distributed":
        logger.error("Distributed mode not implemented yet")
        return 1
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    run_simulation()