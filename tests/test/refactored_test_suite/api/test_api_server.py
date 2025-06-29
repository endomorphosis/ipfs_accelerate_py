#!/usr/bin/env python3
"""
Test API Server

This module provides a FastAPI server for running tests, checking status,
and retrieving results for the IPFS Accelerate Test Suite.
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("test_api_server.log")]
)
logger = logging.getLogger("test_api_server")

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FastAPI components
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Query, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    logger.error("Error: FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# Import test runner
try:
    from test_runner import TestRunner
except ImportError:
    logger.error("Error: TestRunner not found. Make sure test_runner.py is in the same directory.")
    sys.exit(1)

# Import database integration if available
try:
    from database.db_integration import TestDatabaseIntegration
    from database.api_endpoints import init_api as init_db_api
    DATABASE_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Database integration not available. Using memory-only storage.")
    DATABASE_INTEGRATION_AVAILABLE = False

# Define API models
class TestRunRequest(BaseModel):
    """Request model for running a test."""
    model_name: str
    hardware: List[str] = ["cpu"]
    test_type: str = "basic"  # basic, comprehensive, fault_tolerance
    timeout: Optional[int] = 300
    save_results: bool = True

class TestRunResponse(BaseModel):
    """Response model for test run requests."""
    run_id: str
    status: str
    message: str
    started_at: str

class TestAPIServer:
    """
    API server for test execution and management.
    
    This class provides methods for running tests, checking status,
    and retrieving results through a consistent API interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, db_path: Optional[str] = None):
        """Initialize the API server.
        
        Args:
            config: Optional configuration dictionary
            db_path: Optional path to the database file
        """
        self.config = config or {}
        self.results_dir = self.config.get("results_dir", "./test_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize the test runner
        self.runner = TestRunner(config={
            "results_dir": self.results_dir
        })
        
        # WebSocket connections for real-time updates
        self.ws_connections = {}
        
        # Initialize database integration if available
        self.db_integration = None
        if DATABASE_INTEGRATION_AVAILABLE:
            try:
                self.db_integration = TestDatabaseIntegration(db_path)
                logger.info(f"Database integration initialized")
            except Exception as e:
                logger.error(f"Error initializing database integration: {e}")
                self.db_integration = None
    
    async def run_test(self, 
                      request: TestRunRequest, 
                      background_tasks: BackgroundTasks) -> TestRunResponse:
        """
        Run a test asynchronously.
        
        Args:
            request: Test run request parameters
            background_tasks: FastAPI background tasks object
            
        Returns:
            Response with run ID and status
        """
        # Define WebSocket update callback
        async def on_progress(run_id: str):
            await self._send_ws_update(run_id)
        
        # Start the test
        run_id = await self.runner.run_test(
            model_name=request.model_name,
            hardware=request.hardware,
            test_type=request.test_type,
            timeout=request.timeout,
            save_results=request.save_results,
            on_progress=on_progress
        )
        
        # Get initial status
        status = self.runner.get_test_status(run_id)
        
        # Store in database if available
        if self.db_integration:
            try:
                # Prepare run data
                run_data = {
                    "run_id": run_id,
                    "model_name": request.model_name,
                    "hardware": request.hardware,
                    "test_type": request.test_type,
                    "timeout": request.timeout,
                    "save_results": request.save_results,
                    "status": status["status"],
                    "progress": status["progress"],
                    "current_step": status["current_step"],
                    "started_at": datetime.fromisoformat(status["started_at"]) if isinstance(status["started_at"], str) else status["started_at"]
                }
                
                # Track test start
                self.db_integration.track_test_start(run_data)
                logger.info(f"Tracked test start for run {run_id} in database")
            except Exception as e:
                logger.error(f"Error tracking test start in database: {e}")
        
        # Return the response
        return TestRunResponse(
            run_id=run_id,
            status=status["status"],
            message=f"Test run started for {request.model_name}",
            started_at=status["started_at"]
        )
    
    async def _send_ws_update(self, run_id: str):
        """
        Send an update to all WebSocket connections for a run.
        
        Args:
            run_id: The ID of the run that was updated
        """
        if run_id not in self.ws_connections:
            return
            
        # Get the current status
        try:
            status_data = self.runner.get_test_status(run_id)
            
            # Track update in database if available
            if self.db_integration and status_data:
                try:
                    self.db_integration.track_test_update(
                        run_id=run_id,
                        status=status_data["status"],
                        progress=status_data["progress"],
                        current_step=status_data["current_step"],
                        error=status_data.get("error")
                    )
                except Exception as e:
                    logger.error(f"Error tracking test update in database: {e}")
            
            # Send to all connected clients
            for connection in self.ws_connections[run_id]:
                try:
                    await connection.send_json(status_data)
                except Exception as e:
                    logger.error(f"Error sending WebSocket update: {e}")
        except Exception as e:
            logger.error(f"Error getting status for WebSocket update: {e}")
    
    def get_test_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get the status of a test run.
        
        Args:
            run_id: The ID of the run
            
        Returns:
            Dict containing run status information
            
        Raises:
            HTTPException: If the run doesn't exist
        """
        try:
            return self.runner.get_test_status(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")
    
    def get_test_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a completed test run.
        
        Args:
            run_id: The ID of the run
            
        Returns:
            Dict containing run results
            
        Raises:
            HTTPException: If the run doesn't exist or isn't completed
        """
        try:
            # Get results from runner
            results = self.runner.get_test_results(run_id)
            
            # Track completion in database if available
            if self.db_integration and results and results["status"] == "completed":
                try:
                    self.db_integration.track_test_completion(run_id, results)
                    logger.info(f"Tracked test completion for run {run_id} in database")
                except Exception as e:
                    logger.error(f"Error tracking test completion in database: {e}")
            
            return results
            
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Test run {run_id} is not completed")
    
    def list_test_runs(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List recent test runs.
        
        Args:
            limit: Maximum number of runs to return
            status: Optional status filter
            
        Returns:
            List of test run information
        """
        return self.runner.list_test_runs(limit, status)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models for testing.
        
        Returns:
            List of model information
        """
        return self.runner.get_available_models()
    
    def get_available_hardware(self) -> List[Dict[str, Any]]:
        """
        Get a list of available hardware platforms for testing.
        
        Returns:
            List of hardware information
        """
        return self.runner.get_available_hardware()
    
    def get_test_types(self) -> List[Dict[str, Any]]:
        """
        Get a list of available test types.
        
        Returns:
            List of test type information
        """
        return self.runner.get_test_types()
    
    def cancel_test_run(self, run_id: str) -> bool:
        """
        Cancel a running test.
        
        Args:
            run_id: The ID of the test run to cancel
            
        Returns:
            True if the test was cancelled, False otherwise
            
        Raises:
            HTTPException: If the run_id is not found
        """
        try:
            # Cancel the test
            cancelled = self.runner.cancel_test_run(run_id)
            
            # Track cancellation in database if available
            if self.db_integration and cancelled:
                try:
                    self.db_integration.track_test_cancellation(run_id)
                    logger.info(f"Tracked test cancellation for run {run_id} in database")
                except Exception as e:
                    logger.error(f"Error tracking test cancellation in database: {e}")
            
            return cancelled
            
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")
    
    async def handle_websocket_connection(self, websocket: WebSocket, run_id: str):
        """
        Handle a WebSocket connection for a test run.
        
        Args:
            websocket: The WebSocket connection
            run_id: The ID of the run to monitor
        """
        await websocket.accept()
        
        # Check if the run exists
        try:
            status = self.get_test_status(run_id)
        except HTTPException:
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
                elif message == "cancel":
                    cancelled = self.cancel_test_run(run_id)
                    await websocket.send_json({"cancelled": cancelled})
                    
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

# Create the FastAPI application
app = FastAPI(
    title="Test API Server",
    description="API server for running tests and retrieving results",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the test API manager
manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global manager
    
    # Parse arguments (only during direct execution, not when imported)
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Test API Server")
        parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
        parser.add_argument("--results-dir", type=str, default="./test_results", help="Directory for test results")
        parser.add_argument("--db-path", type=str, help="Path to the DuckDB database file")
        args = parser.parse_args()
        
        # Create the test API manager with arguments
        manager = TestAPIServer(
            config={"results_dir": args.results_dir},
            db_path=args.db_path
        )
    else:
        # Default configuration when imported as a module
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "data", "test_runs.duckdb")
        manager = TestAPIServer(db_path=db_path)
    
    # Initialize database API endpoints if available
    if DATABASE_INTEGRATION_AVAILABLE:
        try:
            init_db_api(app)
            logger.info("Database API endpoints initialized")
        except Exception as e:
            logger.error(f"Error initializing database API endpoints: {e}")
    
    logger.info("Test API Server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Close database connection if available
    if manager and manager.db_integration:
        try:
            manager.db_integration.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    logger.info("Test API Server stopped")

@app.post("/api/test/run", response_model=TestRunResponse)
async def run_test(
    request: TestRunRequest,
    background_tasks: BackgroundTasks
):
    """
    Run a test on a model.
    
    Request body parameters:
    - **model_name**: Name of the model to test
    - **hardware**: List of hardware platforms to test on
    - **test_type**: Type of test to run (basic, comprehensive, fault_tolerance)
    - **timeout**: Timeout in seconds
    - **save_results**: Whether to save results to disk
    
    Returns the run ID and status.
    """
    return await manager.run_test(request, background_tasks)

@app.get("/api/test/status/{run_id}")
async def get_test_status(run_id: str):
    """
    Get the status of a test run.
    
    Parameters:
    - **run_id**: The ID of the run to check
    
    Returns the current status of the test run.
    """
    return manager.get_test_status(run_id)

@app.get("/api/test/results/{run_id}")
async def get_test_results(run_id: str):
    """
    Get the results of a completed test run.
    
    Parameters:
    - **run_id**: The ID of the run to get results for
    
    Returns the results of the test run.
    """
    return manager.get_test_results(run_id)

@app.get("/api/test/runs")
async def list_test_runs(
    limit: int = Query(100, gt=0, le=1000),
    status: Optional[str] = None
):
    """
    List recent test runs.
    
    Parameters:
    - **limit**: Maximum number of runs to return
    - **status**: Optional status filter
    
    Returns a list of test runs.
    """
    return manager.list_test_runs(limit, status)

@app.get("/api/test/models")
async def get_models():
    """
    Get a list of available models for testing.
    
    Returns a list of model information.
    """
    return manager.get_available_models()

@app.get("/api/test/hardware")
async def get_hardware():
    """
    Get a list of available hardware platforms for testing.
    
    Returns a list of hardware information.
    """
    return manager.get_available_hardware()

@app.get("/api/test/types")
async def get_test_types():
    """
    Get a list of available test types.
    
    Returns a list of test type information.
    """
    return manager.get_test_types()

@app.post("/api/test/cancel/{run_id}")
async def cancel_test(run_id: str):
    """
    Cancel a running test.
    
    Parameters:
    - **run_id**: The ID of the run to cancel
    
    Returns whether the test was cancelled.
    """
    cancelled = manager.cancel_test_run(run_id)
    return {"run_id": run_id, "cancelled": cancelled}

@app.websocket("/api/test/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time test updates.
    
    Parameters:
    - **run_id**: The ID of the run to monitor
    
    Returns real-time updates on the test progress.
    """
    await manager.handle_websocket_connection(websocket, run_id)

def main():
    """Main entry point when run directly."""
    import uvicorn
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--results-dir", type=str, default="./test_results", help="Directory for test results")
    parser.add_argument("--db-path", type=str, default="./data/test_runs.duckdb", help="Path to the DuckDB database file")
    args = parser.parse_args()
    
    # Start the server
    logger.info(f"Starting Test API Server on {args.host}:{args.port}")
    logger.info(f"Using results directory: {args.results_dir}")
    logger.info(f"Using database path: {args.db_path}")
    
    uvicorn.run(
        "test_api_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()