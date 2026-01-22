#!/usr/bin/env python3
"""
Test Runner Integration for FastAPI

This module provides the implementation bridge between the FastAPI endpoints
and the underlying test execution functionality. It handles the execution of
tests, status tracking, and result management.
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRunner:
    """
    Manages test execution and status tracking.
    
    This class serves as the integration layer between the FastAPI endpoints
    and the underlying test execution functionality. It handles starting test
    runs, tracking their status, and managing results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the test runner.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.active_tests = {}
        self.results_dir = self.config.get("results_dir", "./test_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Try to import test suite components if available
        try:
            # Import from refactored test suite if available
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model_test_base import BaseModelTest
            self.test_base_available = True
            self.BaseModelTest = BaseModelTest
        except ImportError:
            logger.warning("BaseModelTest not available, using simulation mode")
            self.test_base_available = False
    
    async def run_test(self, 
                      model_name: str, 
                      hardware: List[str], 
                      test_type: str,
                      timeout: int = 300,
                      save_results: bool = True,
                      on_progress: Optional[Callable] = None) -> str:
        """
        Run a test asynchronously.
        
        Args:
            model_name: Name of the model to test
            hardware: List of hardware platforms to test on
            test_type: Type of test to run (basic, comprehensive, fault_tolerance)
            timeout: Timeout in seconds
            save_results: Whether to save results to disk
            on_progress: Optional callback for progress updates
            
        Returns:
            Test run ID
        """
        # Generate a unique run ID
        run_id = str(uuid.uuid4())
        
        # Create run configuration
        self.active_tests[run_id] = {
            "run_id": run_id,
            "status": "initializing",
            "progress": 0.0,
            "current_step": "Setting up test environment",
            "started_at": datetime.now(),
            "completed_at": None,
            "model_name": model_name,
            "hardware": hardware,
            "test_type": test_type,
            "timeout": timeout,
            "save_results": save_results,
            "results": {},
            "error": None
        }
        
        # Log start of run
        logger.info(f"Starting test run {run_id} for model {model_name}")
        
        # Start the test in a background task
        asyncio.create_task(
            self._run_test_task(
                run_id=run_id,
                model_name=model_name,
                hardware=hardware,
                test_type=test_type,
                timeout=timeout,
                save_results=save_results,
                on_progress=on_progress
            )
        )
        
        return run_id
    
    async def _run_test_task(self,
                           run_id: str,
                           model_name: str,
                           hardware: List[str],
                           test_type: str,
                           timeout: int,
                           save_results: bool,
                           on_progress: Optional[Callable] = None):
        """
        Execute the test in a background task.
        
        Args:
            run_id: Unique ID for this run
            model_name: Name of the model to test
            hardware: List of hardware platforms to test on
            test_type: Type of test to run
            timeout: Timeout in seconds
            save_results: Whether to save results to disk
            on_progress: Optional callback for progress updates
        """
        try:
            # Update status
            self.active_tests[run_id]["status"] = "running"
            self.active_tests[run_id]["progress"] = 0.1
            self.active_tests[run_id]["current_step"] = "Preparing test environment"
            if on_progress:
                await on_progress(run_id)
            
            # If BaseModelTest is available, use it
            if self.test_base_available:
                # Run the actual test using BaseModelTest
                await self._run_with_base_model_test(
                    run_id, model_name, hardware, test_type, timeout, on_progress
                )
            else:
                # Simulate test execution
                await self._simulate_test_execution(
                    run_id, model_name, hardware, test_type, timeout, on_progress
                )
            
            # Save results if requested
            if save_results:
                result_file = os.path.join(self.results_dir, f"{run_id}.json")
                with open(result_file, 'w') as f:
                    json.dump(
                        self.get_test_results(run_id),
                        f,
                        indent=2,
                        default=str
                    )
                
                self.active_tests[run_id]["result_file"] = result_file
            
            # Update status to completed
            self.active_tests[run_id]["status"] = "completed"
            self.active_tests[run_id]["progress"] = 1.0
            self.active_tests[run_id]["current_step"] = "Test completed"
            self.active_tests[run_id]["completed_at"] = datetime.now()
            if on_progress:
                await on_progress(run_id)
            
            logger.info(f"Test run {run_id} completed successfully")
            
        except asyncio.TimeoutError:
            # Handle timeout
            logger.error(f"Test run {run_id} timed out after {timeout} seconds")
            self.active_tests[run_id]["status"] = "timeout"
            self.active_tests[run_id]["error"] = f"Test timed out after {timeout} seconds"
            self.active_tests[run_id]["current_step"] = "Timeout occurred"
            self.active_tests[run_id]["completed_at"] = datetime.now()
            if on_progress:
                await on_progress(run_id)
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error in test run {run_id}: {str(e)}", exc_info=True)
            self.active_tests[run_id]["status"] = "error"
            self.active_tests[run_id]["error"] = str(e)
            self.active_tests[run_id]["current_step"] = "Error during test execution"
            self.active_tests[run_id]["completed_at"] = datetime.now()
            if on_progress:
                await on_progress(run_id)
    
    async def _run_with_base_model_test(self,
                                      run_id: str,
                                      model_name: str,
                                      hardware: List[str],
                                      test_type: str,
                                      timeout: int,
                                      on_progress: Optional[Callable] = None):
        """
        Run the test using BaseModelTest.
        
        Args:
            run_id: Unique ID for this run
            model_name: Name of the model to test
            hardware: List of hardware platforms to test on
            test_type: Type of test to run
            timeout: Timeout in seconds
            on_progress: Optional callback for progress updates
        """
        # Create test instance
        test_instance = self.BaseModelTest(model_name)
        
        # Configure test based on test_type
        if test_type == "basic":
            test_methods = ["test_load_model", "test_basic_inference"]
        elif test_type == "comprehensive":
            test_methods = ["test_load_model", "test_basic_inference", "test_batch_inference", 
                           "test_model_attributes", "test_save_load"]
        elif test_type == "fault_tolerance":
            test_methods = ["test_load_model", "test_basic_inference", "test_error_handling",
                           "test_resource_cleanup"]
        else:
            # Default to basic tests
            test_methods = ["test_load_model", "test_basic_inference"]
        
        # Initialize results
        results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "test_details": []
        }
        
        # Configure test for hardware
        test_instance.configure_hardware(hardware[0])  # Use first hardware for now
        
        # Setup progress tracking
        total_methods = len(test_methods)
        completed_methods = 0
        
        # Run each test method
        for method_name in test_methods:
            # Update progress
            self.active_tests[run_id]["current_step"] = f"Running {method_name}"
            if on_progress:
                await on_progress(run_id)
            
            # Run the test method
            try:
                # Get the method
                method = getattr(test_instance, method_name)
                
                # Run with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(method),
                    timeout=timeout
                )
                
                # Record success
                results["tests_passed"] += 1
                results["test_details"].append({
                    "method": method_name,
                    "status": "passed",
                    "duration_ms": result.get("duration_ms", 0) if isinstance(result, dict) else 0
                })
                
            except Exception as e:
                # Record failure
                results["tests_failed"] += 1
                results["test_details"].append({
                    "method": method_name,
                    "status": "failed",
                    "error": str(e)
                })
            
            # Update progress
            completed_methods += 1
            progress = 0.1 + (0.9 * (completed_methods / total_methods))
            self.active_tests[run_id]["progress"] = progress
            if on_progress:
                await on_progress(run_id)
        
        # Store results
        self.active_tests[run_id]["results"] = results
    
    async def _simulate_test_execution(self,
                                     run_id: str,
                                     model_name: str,
                                     hardware: List[str],
                                     test_type: str,
                                     timeout: int,
                                     on_progress: Optional[Callable] = None):
        """
        Simulate test execution for development/testing.
        
        Args:
            run_id: Unique ID for this run
            model_name: Name of the model to test
            hardware: List of hardware platforms to test on
            test_type: Type of test to run
            timeout: Timeout in seconds
            on_progress: Optional callback for progress updates
        """
        # Determine number of steps based on test_type
        if test_type == "basic":
            steps = ["test_load_model", "test_basic_inference"]
        elif test_type == "comprehensive":
            steps = ["test_load_model", "test_basic_inference", "test_batch_inference", 
                    "test_model_attributes", "test_save_load"]
        elif test_type == "fault_tolerance":
            steps = ["test_load_model", "test_basic_inference", "test_error_handling",
                    "test_resource_cleanup"]
        else:
            # Default to basic tests
            steps = ["test_load_model", "test_basic_inference"]
        
        # Initialize results
        results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "test_details": []
        }
        
        # Setup progress tracking
        total_steps = len(steps)
        
        # Simulate test execution for each step
        for i, step in enumerate(steps):
            # Update status
            progress = 0.1 + (0.9 * (i / total_steps))
            self.active_tests[run_id]["progress"] = progress
            self.active_tests[run_id]["current_step"] = f"Running {step}"
            if on_progress:
                await on_progress(run_id)
            
            # Simulate work
            await asyncio.sleep(1)
            
            # Simulate test result (95% pass rate for demonstration)
            import random
            if random.random() < 0.95:
                # Test passed
                results["tests_passed"] += 1
                results["test_details"].append({
                    "method": step,
                    "status": "passed",
                    "duration_ms": random.randint(50, 500)
                })
            else:
                # Test failed
                results["tests_failed"] += 1
                results["test_details"].append({
                    "method": step,
                    "status": "failed",
                    "error": "Simulated test failure"
                })
        
        # For hardware benchmarking, add performance metrics for each hardware type
        performance_metrics = {}
        for hw in hardware:
            performance_metrics[hw] = {
                "latency_ms": round(10 + random.random() * 90, 2),
                "throughput_items_per_sec": round(random.random() * 100, 2),
                "memory_mb": round(100 + random.random() * 900, 2)
            }
        
        # Store results
        results["performance_metrics"] = performance_metrics
        self.active_tests[run_id]["results"] = results
    
    def get_test_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get the current status of a test run.
        
        Args:
            run_id: The ID of the test run
            
        Returns:
            Dict containing the test status
            
        Raises:
            KeyError: If the run_id is not found
        """
        if run_id not in self.active_tests:
            raise KeyError(f"Test run {run_id} not found")
        
        test_data = self.active_tests[run_id]
        
        # Calculate elapsed time
        elapsed = (datetime.now() - test_data["started_at"]).total_seconds()
        
        # Calculate estimated remaining time
        remaining = None
        if test_data["status"] == "running" and test_data["progress"] > 0:
            remaining = (elapsed / test_data["progress"]) * (1.0 - test_data["progress"])
        
        # Prepare status response
        status = {
            "run_id": run_id,
            "status": test_data["status"],
            "progress": test_data["progress"],
            "current_step": test_data["current_step"],
            "model_name": test_data["model_name"],
            "hardware": test_data["hardware"],
            "test_type": test_data["test_type"],
            "started_at": test_data["started_at"].isoformat(),
            "elapsed_time": elapsed,
            "estimated_remaining_time": remaining
        }
        
        # Include error if available
        if test_data.get("error"):
            status["error"] = test_data["error"]
        
        return status
    
    def get_test_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a completed test run.
        
        Args:
            run_id: The ID of the test run
            
        Returns:
            Dict containing the test results
            
        Raises:
            KeyError: If the run_id is not found
            ValueError: If the test is not completed
        """
        if run_id not in self.active_tests:
            raise KeyError(f"Test run {run_id} not found")
        
        test_data = self.active_tests[run_id]
        
        if test_data["status"] not in ["completed", "error", "timeout"]:
            raise ValueError(f"Test run {run_id} is not completed")
        
        # Prepare results response
        results = {
            "run_id": run_id,
            "status": test_data["status"],
            "model_name": test_data["model_name"],
            "hardware": test_data["hardware"],
            "test_type": test_data["test_type"],
            "started_at": test_data["started_at"].isoformat(),
            "completed_at": test_data["completed_at"].isoformat() if test_data.get("completed_at") else None,
            "duration": (test_data.get("completed_at", datetime.now()) - test_data["started_at"]).total_seconds(),
            "results": test_data.get("results", {})
        }
        
        # Include error if available
        if test_data.get("error"):
            results["error"] = test_data["error"]
        
        # Include result file if available
        if test_data.get("result_file"):
            results["result_file"] = test_data["result_file"]
        
        return results
    
    def list_test_runs(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List recent test runs.
        
        Args:
            limit: Maximum number of runs to return
            status: Optional status filter
            
        Returns:
            List of test run information
        """
        runs = []
        
        # Get all run IDs
        run_ids = list(self.active_tests.keys())
        
        # Sort by start time (most recent first)
        run_ids.sort(key=lambda x: self.active_tests[x]["started_at"], reverse=True)
        
        # Apply status filter if provided
        if status:
            run_ids = [r for r in run_ids if self.active_tests[r]["status"] == status]
        
        # Apply limit
        run_ids = run_ids[:limit]
        
        # Prepare response
        for run_id in run_ids:
            test_data = self.active_tests[run_id]
            runs.append({
                "run_id": run_id,
                "status": test_data["status"],
                "model_name": test_data["model_name"],
                "hardware": test_data["hardware"],
                "test_type": test_data["test_type"],
                "started_at": test_data["started_at"].isoformat(),
                "completed_at": test_data["completed_at"].isoformat() if test_data.get("completed_at") else None
            })
        
        return runs
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models for testing.
        
        Returns:
            List of model information
        """
        # In a real implementation, this would query the model registry
        # For now, return a static list of common models
        return [
            {"name": "bert-base-uncased", "type": "encoder", "modality": "text"},
            {"name": "gpt2", "type": "decoder", "modality": "text"},
            {"name": "t5-small", "type": "encoder-decoder", "modality": "text"},
            {"name": "vit-base-patch16-224", "type": "encoder", "modality": "vision"},
            {"name": "clip-vit-base-patch32", "type": "multimodal", "modality": "vision-text"},
            {"name": "whisper-tiny", "type": "encoder-decoder", "modality": "audio"}
        ]
    
    def get_available_hardware(self) -> List[Dict[str, Any]]:
        """
        Get a list of available hardware platforms for testing.
        
        Returns:
            List of hardware information
        """
        # In a real implementation, this would query the hardware manager
        # For now, return a static list of common hardware platforms
        return [
            {"name": "cpu", "available": True, "type": "CPU", "description": "CPU execution"},
            {"name": "cuda", "available": True, "type": "GPU", "description": "NVIDIA CUDA GPU acceleration"},
            {"name": "rocm", "available": False, "type": "GPU", "description": "AMD ROCm GPU acceleration"},
            {"name": "openvino", "available": True, "type": "CPU", "description": "Intel OpenVINO acceleration"},
            {"name": "webgpu", "available": True, "type": "WebGPU", "description": "WebGPU acceleration in browser"},
            {"name": "webnn", "available": True, "type": "WebNN", "description": "WebNN acceleration in browser"}
        ]
    
    def get_test_types(self) -> List[Dict[str, Any]]:
        """
        Get a list of available test types.
        
        Returns:
            List of test type information
        """
        return [
            {
                "id": "basic",
                "name": "Basic Tests",
                "description": "Basic model loading and inference tests",
                "methods": ["test_load_model", "test_basic_inference"]
            },
            {
                "id": "comprehensive",
                "name": "Comprehensive Tests",
                "description": "Thorough testing of model capabilities and performance",
                "methods": ["test_load_model", "test_basic_inference", "test_batch_inference", 
                           "test_model_attributes", "test_save_load"]
            },
            {
                "id": "fault_tolerance",
                "name": "Fault Tolerance Tests",
                "description": "Tests for error handling and resource management",
                "methods": ["test_load_model", "test_basic_inference", "test_error_handling",
                           "test_resource_cleanup"]
            }
        ]

    def cancel_test_run(self, run_id: str) -> bool:
        """
        Cancel a running test.
        
        Args:
            run_id: The ID of the test run to cancel
            
        Returns:
            True if the test was cancelled, False otherwise
            
        Raises:
            KeyError: If the run_id is not found
        """
        if run_id not in self.active_tests:
            raise KeyError(f"Test run {run_id} not found")
        
        test_data = self.active_tests[run_id]
        
        # Only running tests can be cancelled
        if test_data["status"] != "running":
            return False
        
        # Update status
        test_data["status"] = "cancelled"
        test_data["current_step"] = "Test cancelled by user"
        test_data["completed_at"] = datetime.now()
        
        logger.info(f"Test run {run_id} cancelled")
        return True
    
    def cleanup_old_runs(self, max_age_days: int = 7) -> int:
        """
        Clean up old test runs from memory.
        
        Args:
            max_age_days: Maximum age of test runs to keep
            
        Returns:
            Number of cleaned up runs
        """
        threshold = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        runs_to_remove = []
        
        for run_id, test_data in self.active_tests.items():
            # Skip running tests
            if test_data["status"] == "running":
                continue
                
            # Check if the test is old enough to remove
            if test_data["started_at"].timestamp() < threshold:
                runs_to_remove.append(run_id)
        
        # Remove the runs
        for run_id in runs_to_remove:
            del self.active_tests[run_id]
        
        return len(runs_to_remove)
    
    def clear_test_results(self, run_id: str) -> bool:
        """
        Clear the results of a completed test run to save memory.
        
        Args:
            run_id: The ID of the test run
            
        Returns:
            True if the results were cleared, False otherwise
            
        Raises:
            KeyError: If the run_id is not found
        """
        if run_id not in self.active_tests:
            raise KeyError(f"Test run {run_id} not found")
        
        test_data = self.active_tests[run_id]
        
        # Only completed tests can have their results cleared
        if test_data["status"] not in ["completed", "error", "timeout", "cancelled"]:
            return False
        
        # Clear the results but keep the metadata
        if "results" in test_data:
            # Keep test summary but remove details
            if isinstance(test_data["results"], dict):
                summary = {
                    "tests_passed": test_data["results"].get("tests_passed", 0),
                    "tests_failed": test_data["results"].get("tests_failed", 0),
                    "tests_skipped": test_data["results"].get("tests_skipped", 0)
                }
                test_data["results"] = summary
            else:
                test_data["results"] = {}
                
        return True