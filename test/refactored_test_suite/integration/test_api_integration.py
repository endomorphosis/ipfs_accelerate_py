#!/usr/bin/env python3
"""
Integration Test for Test Suite API Integration

This module provides integration tests for verifying the FastAPI interfaces 
that connect the test suite to external components. It ensures consistent API
patterns across the refactored components.

Example usage:
    python -m refactored_test_suite.integration.test_api_integration
"""

import os
import sys
import unittest
import json
import logging
import tempfile
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directories to path to allow imports
current_dir = Path(__file__).resolve().parent
test_dir = current_dir.parent.parent
sys.path.append(str(test_dir))

# Import required modules
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not available, some tests will be skipped")
    FASTAPI_AVAILABLE = False

# Import from refactored test suite
try:
    from refactored_test_suite.model_test_base import BaseModelTest
    TEST_SUITE_AVAILABLE = True
except ImportError:
    logger.warning("Test suite components not available, some tests will be skipped")
    TEST_SUITE_AVAILABLE = False

# Import from refactored benchmark suite
try:
    from refactored_benchmark_suite.benchmark_core.runner import BenchmarkRunner
    BENCHMARK_AVAILABLE = True
except ImportError:
    logger.warning("Benchmark components not available, some tests will be skipped")
    BENCHMARK_AVAILABLE = False

# Import from refactored generator suite
try:
    from refactored_generator_suite.generator_core.generator import ModelGenerator
    GENERATOR_AVAILABLE = True
except ImportError:
    logger.warning("Generator components not available, some tests will be skipped")
    GENERATOR_AVAILABLE = False

# Define API models
class TestRunRequest(BaseModel):
    """Request model for running tests"""
    model_name: str
    hardware: List[str] = ["cpu"]
    test_type: str = "basic"  # basic, comprehensive, fault_tolerance
    timeout: Optional[int] = 300
    save_results: bool = True

class TestRunResponse(BaseModel):
    """Response model for test run requests"""
    run_id: str
    status: str
    message: str
    started_at: str

class TestStatus(BaseModel):
    """Status of a test run"""
    run_id: str
    status: str
    progress: float
    current_step: str
    started_at: str
    elapsed_time: float
    estimated_remaining_time: Optional[float] = None
    error: Optional[str] = None

class TestResult(BaseModel):
    """Results of a test run"""
    run_id: str
    model_name: str
    hardware: List[str]
    test_type: str
    status: str
    results: Dict[str, Any]
    started_at: str
    completed_at: Optional[str] = None
    duration: Optional[float] = None

class GenerateModelRequest(BaseModel):
    """Request model for generating model code"""
    model_name: str
    hardware: List[str] = ["cpu"]
    output_dir: Optional[str] = None
    force: bool = False
    template_type: Optional[str] = None

class GenerateModelResponse(BaseModel):
    """Response model for model generation requests"""
    task_id: str
    status: str
    message: str
    file_path: Optional[str] = None

class TestAPIServer:
    """
    API server for test execution and management.
    
    This class provides methods for running tests, checking status,
    and retrieving results through a consistent API interface.
    """
    
    def __init__(self):
        """Initialize the API server"""
        self.active_tests = {}
        self.results_dir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.ws_connections = {}
    
    async def run_test(self, 
                      request: TestRunRequest, 
                      background_tasks: BackgroundTasks) -> TestRunResponse:
        """
        Run a test asynchronously
        
        Args:
            request: Test run request parameters
            background_tasks: FastAPI background tasks object
            
        Returns:
            Response with run ID and status
        """
        # Generate a unique run ID
        import uuid
        from datetime import datetime
        
        run_id = str(uuid.uuid4())
        
        # Create run configuration
        self.active_tests[run_id] = {
            "run_id": run_id,
            "status": "initializing",
            "progress": 0.0,
            "current_step": "Setting up test environment",
            "started_at": datetime.now(),
            "model_name": request.model_name,
            "hardware": request.hardware,
            "test_type": request.test_type,
            "timeout": request.timeout,
            "save_results": request.save_results,
            "results": {}
        }
        
        # Log start of run
        logger.info(f"Starting test run {run_id} for model {request.model_name}")
        
        # Add the task to run asynchronously
        background_tasks.add_task(
            self._run_test,
            run_id=run_id,
            model_name=request.model_name,
            hardware=request.hardware,
            test_type=request.test_type,
            timeout=request.timeout,
            save_results=request.save_results
        )
        
        # Return the response
        return TestRunResponse(
            run_id=run_id,
            status="initializing",
            message=f"Test run started for {request.model_name}",
            started_at=self.active_tests[run_id]["started_at"].isoformat()
        )
    
    async def _run_test(self,
                       run_id: str,
                       model_name: str,
                       hardware: List[str],
                       test_type: str,
                       timeout: int,
                       save_results: bool):
        """
        Run the test in the background
        
        Args:
            run_id: Unique ID for this run
            model_name: Name of the model to test
            hardware: List of hardware to test on
            test_type: Type of test to run
            timeout: Timeout in seconds
            save_results: Whether to save results to disk
        """
        try:
            # Update status
            self.active_tests[run_id]["status"] = "running"
            self.active_tests[run_id]["current_step"] = "Initializing test"
            await self._send_ws_update(run_id)
            
            # Simulated test execution - in real implementation this would use BaseModelTest
            await asyncio.sleep(2)  # Simulate setup time
            
            # Update progress
            self.active_tests[run_id]["progress"] = 0.1
            self.active_tests[run_id]["current_step"] = f"Running {test_type} test on {model_name}"
            await self._send_ws_update(run_id)
            
            # Simulate test execution phases
            for hardware_type in hardware:
                self.active_tests[run_id]["current_step"] = f"Testing on {hardware_type}"
                await self._send_ws_update(run_id)
                
                # Simulate workload
                await asyncio.sleep(1)
                progress_increment = 0.9 / len(hardware)
                self.active_tests[run_id]["progress"] += progress_increment
                await self._send_ws_update(run_id)
            
            # Simulate test completion
            self.active_tests[run_id]["status"] = "completed"
            self.active_tests[run_id]["progress"] = 1.0
            self.active_tests[run_id]["current_step"] = "Test completed"
            
            # Generate mock results
            from datetime import datetime
            self.active_tests[run_id]["results"] = {
                "model_name": model_name,
                "hardware": hardware,
                "test_type": test_type,
                "success": True,
                "error": None,
                "details": {
                    "tests_passed": 10,
                    "tests_failed": 0,
                    "tests_skipped": 0
                },
                "completed_at": datetime.now().isoformat()
            }
            
            # Save results if requested
            if save_results:
                result_path = os.path.join(self.results_dir, f"{run_id}.json")
                with open(result_path, 'w') as f:
                    json.dump(self.active_tests[run_id]["results"], f, indent=2)
            
            await self._send_ws_update(run_id)
            logger.info(f"Test run {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error running test {run_id}: {e}", exc_info=True)
            self.active_tests[run_id]["status"] = "failed"
            self.active_tests[run_id]["error"] = str(e)
            self.active_tests[run_id]["current_step"] = "Error running test"
            await self._send_ws_update(run_id)
    
    async def _send_ws_update(self, run_id: str):
        """
        Send an update to all WebSocket connections for a run
        
        Args:
            run_id: The ID of the run that was updated
        """
        if run_id not in self.ws_connections:
            return
            
        # Get the current status
        status_data = self.get_test_status(run_id)
        
        # Send to all connected clients
        for connection in self.ws_connections[run_id]:
            try:
                await connection.send_json(status_data)
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {e}")
    
    def get_test_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get the status of a test run
        
        Args:
            run_id: The ID of the run
            
        Returns:
            Dict containing run status information
            
        Raises:
            HTTPException: If the run doesn't exist
        """
        if run_id not in self.active_tests:
            raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")
            
        test_data = self.active_tests[run_id]
        
        # Calculate elapsed time
        from datetime import datetime
        elapsed = (datetime.now() - test_data["started_at"]).total_seconds()
        
        # Calculate estimated remaining time
        remaining = None
        if test_data["progress"] > 0 and test_data["progress"] < 1.0:
            remaining = (elapsed / test_data["progress"]) * (1.0 - test_data["progress"])
        
        # Create status object
        status = {
            "run_id": run_id,
            "status": test_data["status"],
            "progress": test_data["progress"],
            "current_step": test_data["current_step"],
            "started_at": test_data["started_at"].isoformat(),
            "elapsed_time": elapsed,
            "estimated_remaining_time": remaining
        }
        
        if "error" in test_data:
            status["error"] = test_data["error"]
            
        return status
    
    def get_test_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a completed test run
        
        Args:
            run_id: The ID of the run
            
        Returns:
            Dict containing run results
            
        Raises:
            HTTPException: If the run doesn't exist or isn't completed
        """
        if run_id not in self.active_tests:
            raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")
            
        test_data = self.active_tests[run_id]
        
        if test_data["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Test run {run_id} is not completed")
            
        # Return the results
        from datetime import datetime
        
        return {
            "run_id": run_id,
            "model_name": test_data["model_name"],
            "hardware": test_data["hardware"],
            "test_type": test_data["test_type"],
            "status": test_data["status"],
            "results": test_data["results"],
            "started_at": test_data["started_at"].isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration": (datetime.now() - test_data["started_at"]).total_seconds()
        }
    
    async def handle_websocket_connection(self, websocket: WebSocket, run_id: str):
        """
        Handle a WebSocket connection for a test run
        
        Args:
            websocket: The WebSocket connection
            run_id: The ID of the run to monitor
        """
        await websocket.accept()
        
        # Check if the run exists
        if run_id not in self.active_tests:
            await websocket.send_json({
                "error": f"Test run {run_id} not found"
            })
            await websocket.close()
            return
            
        # Add the connection to the list for this run
        if run_id not in self.ws_connections:
            self.ws_connections[run_id] = []
            
        self.ws_connections[run_id].append(websocket)
        
        try:
            # Send initial status
            status = self.get_test_status(run_id)
            await websocket.send_json(status)
            
            # Keep the connection open and handle messages
            while True:
                message = await websocket.receive_text()
                
                # Handle client messages if needed
                if message == "ping":
                    await websocket.send_json({"pong": True})
                elif message == "status":
                    status = self.get_test_status(run_id)
                    await websocket.send_json(status)
                    
        except WebSocketDisconnect:
            # Remove the connection from the list
            if run_id in self.ws_connections:
                self.ws_connections[run_id].remove(websocket)
                
                # Clean up empty lists
                if not self.ws_connections[run_id]:
                    del self.ws_connections[run_id]
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            
            # Ensure connection is removed on error
            if run_id in self.ws_connections and websocket in self.ws_connections[run_id]:
                self.ws_connections[run_id].remove(websocket)
                
                # Clean up empty lists
                if not self.ws_connections[run_id]:
                    del self.ws_connections[run_id]

# Create FastAPI app for testing
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Test Suite API",
        description="API for running and monitoring tests",
        version="1.0.0"
    )
    
    # Create the test manager
    test_manager = TestAPIServer()
    
    @app.post("/api/test/run", response_model=TestRunResponse)
    async def start_test_run(
        request: TestRunRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Start a new test run
        
        Request body parameters:
        - **model_name**: Name of the model to test
        - **hardware**: List of hardware to test on
        - **test_type**: Type of test to run (basic, comprehensive, fault_tolerance)
        - **timeout**: Timeout in seconds
        - **save_results**: Whether to save results to disk
        
        Returns the run ID and status
        """
        return await test_manager.run_test(request, background_tasks)
    
    @app.get("/api/test/status/{run_id}")
    async def get_test_status(run_id: str):
        """
        Get the status of a test run
        
        Parameters:
        - **run_id**: The ID of the run to check
        
        Returns the current status of the test run
        """
        return test_manager.get_test_status(run_id)
    
    @app.get("/api/test/results/{run_id}")
    async def get_test_results(run_id: str):
        """
        Get the results of a completed test run
        
        Parameters:
        - **run_id**: The ID of the run to get results for
        
        Returns the results of the test run
        """
        return test_manager.get_test_results(run_id)
    
    @app.websocket("/api/test/ws/{run_id}")
    async def websocket_endpoint(websocket: WebSocket, run_id: str):
        """
        WebSocket endpoint for real-time test updates
        
        Parameters:
        - **run_id**: The ID of the run to monitor
        
        Returns real-time updates on the test progress
        """
        await test_manager.handle_websocket_connection(websocket, run_id)

@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPIIntegrationTest(unittest.TestCase):
    """Integration tests for the Test API interface."""
    
    def setUp(self):
        """Set up test client and environment."""
        if FASTAPI_AVAILABLE:
            self.client = TestClient(app)
        
    def test_run_test_endpoint(self):
        """Test the /api/test/run endpoint."""
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI not available")
            
        # Prepare test request
        test_request = {
            "model_name": "bert-base-uncased",
            "hardware": ["cpu"],
            "test_type": "basic",
            "timeout": 60,
            "save_results": True
        }
        
        # Make request
        response = self.client.post("/api/test/run", json=test_request)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("run_id", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "initializing")
        
        # Store run_id for subsequent tests
        self.run_id = data["run_id"]
    
    def test_get_status_endpoint(self):
        """Test the /api/test/status/{run_id} endpoint."""
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI not available")
            
        # Run a test first to get a run_id
        test_request = {
            "model_name": "bert-base-uncased",
            "hardware": ["cpu"],
            "test_type": "basic"
        }
        response = self.client.post("/api/test/run", json=test_request)
        run_id = response.json()["run_id"]
        
        # Get status
        time.sleep(1)  # Give the background task time to start
        status_response = self.client.get(f"/api/test/status/{run_id}")
        
        # Check response
        self.assertEqual(status_response.status_code, 200)
        status_data = status_response.json()
        self.assertEqual(status_data["run_id"], run_id)
        self.assertIn("progress", status_data)
        self.assertIn("current_step", status_data)
    
    def test_get_results_endpoint(self):
        """Test the /api/test/results/{run_id} endpoint."""
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI not available")
            
        # Run a test first to get a run_id
        test_request = {
            "model_name": "bert-base-uncased",
            "hardware": ["cpu"],
            "test_type": "basic"
        }
        response = self.client.post("/api/test/run", json=test_request)
        run_id = response.json()["run_id"]
        
        # Wait for test to complete (simulation takes about 4-5 seconds)
        time.sleep(6)
        
        # Get results
        results_response = self.client.get(f"/api/test/results/{run_id}")
        
        # Check response
        self.assertEqual(results_response.status_code, 200)
        results_data = results_response.json()
        self.assertEqual(results_data["run_id"], run_id)
        self.assertEqual(results_data["status"], "completed")
        self.assertIn("results", results_data)
        self.assertIn("model_name", results_data)
        self.assertEqual(results_data["model_name"], "bert-base-uncased")

@unittest.skipIf(not TEST_SUITE_AVAILABLE or not BENCHMARK_AVAILABLE or not GENERATOR_AVAILABLE,
                "Required components not available")
class FullStackIntegrationTest(unittest.TestCase):
    """Test integration between all refactored components."""
    
    def test_integration_simulation(self):
        """
        Simulated test of the full stack integration.
        
        This is a placeholder that will be implemented as the
        components are fully refactored. Currently demonstrates
        the intended flow without real implementations.
        """
        logger.info("Testing integration between generator, test suite, and benchmark components")
        self.skipTest("Full stack integration test not yet implemented")
        
        # The planned integration flow:
        # 1. Use generator to create model implementation
        # 2. Use test suite to validate the implementation
        # 3. Use benchmark suite to measure performance
        # 4. Use API to expose functionality consistently

def run_api_server():
    """Run the FastAPI server."""
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

def run_tests():
    """Run the integration tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_api_server()
    else:
        run_tests()