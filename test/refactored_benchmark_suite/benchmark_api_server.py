#!/usr/bin/env python3
"""
Benchmark API Server

This module provides a FastAPI server for interacting with the benchmark system,
allowing for benchmark execution, monitoring, and result retrieval through
RESTful APIs and WebSockets.
"""

import os
import sys
import json
import time
import uuid
import logging
import anyio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FastAPI components
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Query, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("Error: FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# Import benchmark components
from benchmark_core.runner import BenchmarkRunner
from benchmark_core.registry import BenchmarkRegistry
from benchmark_core.db_integration import BenchmarkDBManager, BenchmarkDBContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("benchmark_api_server.log")]
)
logger = logging.getLogger("benchmark_api_server")

# Define API models
class BenchmarkRunRequest(BaseModel):
    """Request model for starting a benchmark run."""
    priority: str = "high"  # critical, high, medium, all
    hardware: List[str] = ["cpu"]
    models: Optional[List[str]] = None
    batch_sizes: List[int] = [1, 8]
    precision: str = "fp32"
    progressive_mode: bool = True
    incremental: bool = True
    skillset_dir: Optional[str] = None  # If None, use default test skillset dir

class BenchmarkRunResponse(BaseModel):
    """Response model for benchmark run requests."""
    run_id: str
    status: str
    message: str
    started_at: str

class BenchmarkStatus(BaseModel):
    """Status of a benchmark run."""
    run_id: str
    status: str
    progress: float
    current_step: str
    start_time: str
    elapsed_time: float
    estimated_remaining_time: Optional[float] = None
    completed_models: int
    total_models: int
    error: Optional[str] = None

class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    family: Optional[str] = None
    type: Optional[str] = None
    modality: Optional[str] = None

class HardwarePlatform(BaseModel):
    """Information about a hardware platform."""
    name: str
    device: Optional[str] = None
    memory_gb: Optional[float] = None
    simulation_mode: bool = False

class BenchmarkManager:
    """
    Manager class for handling benchmark operations.
    
    This class provides methods for starting benchmark runs, checking status,
    getting results, and handling WebSocket connections for real-time updates.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the benchmark manager.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.db_path = self.config.get("db_path", "./benchmark_db.duckdb")
        self.active_runs = {}
        self.ws_connections = {}
        self.default_skillset_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "..", 
                "ipfs_accelerate_py", 
                "worker", 
                "skillset"
            )
        )
        
        # Initialize results directory
        self.results_dir = self.config.get("results_dir", "./benchmark_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
    async def start_benchmark_run(self, 
                                  request: BenchmarkRunRequest, 
                                  background_tasks: BackgroundTasks) -> BenchmarkRunResponse:
        """
        Start a new benchmark run asynchronously.
        
        Args:
            request: Benchmark run request parameters
            background_tasks: FastAPI background tasks object
            
        Returns:
            Response with run ID and status
        """
        # Generate a unique run ID
        run_id = str(uuid.uuid4())
        
        # Set skillset directory (use test implementation, not prod)
        skillset_dir = request.skillset_dir or self.default_skillset_dir
        
        # Create run configuration
        self.active_runs[run_id] = {
            "run_id": run_id,
            "status": "initializing",
            "progress": 0.0,
            "current_step": "Setting up benchmark environment",
            "start_time": datetime.now(),
            "config": request.dict(),
            "skillset_dir": skillset_dir,
            "completed_models": 0,
            "total_models": 0,
            "results": {}
        }

        # Log start of run
        logger.info(f"Starting benchmark run {run_id} with configuration: {json.dumps(request.dict())}")
        
        # Add the task to run asynchronously
        background_tasks.add_task(
            self._run_benchmark, 
            run_id=run_id, 
            priority=request.priority,
            hardware=request.hardware,
            models=request.models,
            batch_sizes=request.batch_sizes,
            precision=request.precision,
            progressive_mode=request.progressive_mode,
            incremental=request.incremental,
            skillset_dir=skillset_dir
        )
        
        # Return the response
        return BenchmarkRunResponse(
            run_id=run_id,
            status="initializing",
            message="Benchmark run started",
            started_at=self.active_runs[run_id]["start_time"].isoformat()
        )
    
    async def _run_benchmark(self, 
                             run_id: str, 
                             priority: str, 
                             hardware: List[str], 
                             models: Optional[List[str]],
                             batch_sizes: List[int],
                             precision: str,
                             progressive_mode: bool,
                             incremental: bool,
                             skillset_dir: str):
        """
        Run the benchmark in the background.
        
        Args:
            run_id: Unique ID for this run
            priority: Benchmark priority (critical, high, medium, all)
            hardware: List of hardware to benchmark on
            models: Optional list of specific models to benchmark
            batch_sizes: List of batch sizes to benchmark
            precision: Precision to use (fp32, fp16, etc.)
            progressive_mode: Whether to use progressive complexity mode
            incremental: Whether to run only missing or outdated benchmarks
            skillset_dir: Directory containing skillset implementations
        """
        try:
            # Update status
            self.active_runs[run_id]["status"] = "running"
            self.active_runs[run_id]["current_step"] = "Preparing benchmark environment"
            await self._send_ws_update(run_id)
            
            # Import run_complete_benchmark_pipeline in the background task
            # to avoid loading it during FastAPI startup
            from run_complete_benchmark_pipeline import BenchmarkPipeline

            # Create the benchmark pipeline
            self.active_runs[run_id]["current_step"] = "Creating benchmark pipeline"
            await self._send_ws_update(run_id)
            
            pipeline = BenchmarkPipeline(
                priority=priority,
                hardware_types=hardware,
                batch_sizes=batch_sizes,
                precision=precision,
                progressive_mode=progressive_mode,
                incremental=incremental,
                db_path=self.db_path,
                output_dir=os.path.join(self.results_dir, run_id),
                skillset_dir=skillset_dir,
                specific_models=models,
                on_progress_callback=lambda progress, step, completed, total: 
                    # TODO: Replace with task group - asyncio.create_task(self._update_progress(run_id, progress, step, completed, total))
            )
            
            # Initialize the pipeline
            self.active_runs[run_id]["current_step"] = "Initializing benchmark pipeline"
            await self._send_ws_update(run_id)
            
            # Run model discovery to get total model count
            self.active_runs[run_id]["current_step"] = "Discovering models"
            await self._send_ws_update(run_id)
            
            models_to_run = pipeline.discover_models()
            self.active_runs[run_id]["total_models"] = len(models_to_run)
            await self._send_ws_update(run_id)
            
            # Run the pipeline
            self.active_runs[run_id]["current_step"] = "Running benchmarks"
            await self._send_ws_update(run_id)
            
            results = await anyio.to_thread.run_sync(pipeline.run)
            
            # Update status
            self.active_runs[run_id]["status"] = "completed"
            self.active_runs[run_id]["progress"] = 1.0
            self.active_runs[run_id]["current_step"] = "Benchmark completed"
            self.active_runs[run_id]["results"] = results
            await self._send_ws_update(run_id)
            
            logger.info(f"Benchmark run {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error running benchmark {run_id}: {e}", exc_info=True)
            self.active_runs[run_id]["status"] = "failed"
            self.active_runs[run_id]["error"] = str(e)
            self.active_runs[run_id]["current_step"] = "Error running benchmark"
            await self._send_ws_update(run_id)
    
    async def _update_progress(self, 
                              run_id: str, 
                              progress: float, 
                              step: str, 
                              completed_models: int, 
                              total_models: int):
        """
        Update the progress of a benchmark run.
        
        Args:
            run_id: The ID of the run to update
            progress: The progress value (0.0 to 1.0)
            step: The current step description
            completed_models: Number of completed models
            total_models: Total number of models to benchmark
        """
        if run_id not in self.active_runs:
            return
            
        self.active_runs[run_id]["progress"] = progress
        self.active_runs[run_id]["current_step"] = step
        self.active_runs[run_id]["completed_models"] = completed_models
        self.active_runs[run_id]["total_models"] = total_models or self.active_runs[run_id]["total_models"]
        
        await self._send_ws_update(run_id)
    
    async def _send_ws_update(self, run_id: str):
        """
        Send an update to all WebSocket connections for a run.
        
        Args:
            run_id: The ID of the run that was updated
        """
        if run_id not in self.ws_connections:
            return
            
        # Get the current status
        status_data = self.get_run_status(run_id)
        
        # Send to all connected clients
        for connection in self.ws_connections[run_id]:
            try:
                await connection.send_json(status_data)
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {e}")
    
    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get the status of a benchmark run.
        
        Args:
            run_id: The ID of the run
            
        Returns:
            Dict containing run status information
            
        Raises:
            HTTPException: If the run doesn't exist
        """
        if run_id not in self.active_runs:
            raise HTTPException(status_code=404, detail=f"Benchmark run {run_id} not found")
            
        run_data = self.active_runs[run_id]
        
        # Calculate elapsed and estimated remaining time
        elapsed = (datetime.now() - run_data["start_time"]).total_seconds()
        
        remaining = None
        if run_data["progress"] > 0 and run_data["progress"] < 1.0:
            remaining = (elapsed / run_data["progress"]) * (1.0 - run_data["progress"])
        
        # Create status object
        status = {
            "run_id": run_id,
            "status": run_data["status"],
            "progress": run_data["progress"],
            "current_step": run_data["current_step"],
            "start_time": run_data["start_time"].isoformat(),
            "elapsed_time": elapsed,
            "estimated_remaining_time": remaining,
            "completed_models": run_data["completed_models"],
            "total_models": run_data["total_models"]
        }
        
        if "error" in run_data:
            status["error"] = run_data["error"]
            
        return status
    
    def get_run_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a completed benchmark run.
        
        Args:
            run_id: The ID of the run
            
        Returns:
            Dict containing run results
            
        Raises:
            HTTPException: If the run doesn't exist or isn't completed
        """
        if run_id not in self.active_runs:
            raise HTTPException(status_code=404, detail=f"Benchmark run {run_id} not found")
            
        run_data = self.active_runs[run_id]
        
        if run_data["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Benchmark run {run_id} is not completed")
            
        # Return the results
        return {
            "run_id": run_id,
            "status": run_data["status"],
            "config": run_data["config"],
            "results": run_data["results"],
            "start_time": run_data["start_time"].isoformat(),
            "completion_time": datetime.now().isoformat() if "completed_at" not in run_data else run_data["completed_at"].isoformat()
        }
    
    async def handle_websocket_connection(self, websocket: WebSocket, run_id: str):
        """
        Handle a WebSocket connection for a benchmark run.
        
        Args:
            websocket: The WebSocket connection
            run_id: The ID of the run to monitor
        """
        await websocket.accept()
        
        # Check if the run exists
        if run_id not in self.active_runs:
            await websocket.send_json({
                "error": f"Benchmark run {run_id} not found"
            })
            await websocket.close()
            return
            
        # Add the connection to the list for this run
        if run_id not in self.ws_connections:
            self.ws_connections[run_id] = []
            
        self.ws_connections[run_id].append(websocket)
        
        try:
            # Send initial status
            status = self.get_run_status(run_id)
            await websocket.send_json(status)
            
            # Keep the connection open and handle messages
            while True:
                message = await websocket.receive_text()
                
                # Handle client messages if needed
                if message == "ping":
                    await websocket.send_json({"pong": True})
                elif message == "status":
                    status = self.get_run_status(run_id)
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
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get a list of available models for benchmarking.
        
        Returns:
            List of model information
        """
        try:
            # Load model family classification module
            from benchmark_core.huggingface_integration import ModelArchitectureRegistry
            
            # Get all registered model families
            model_registry = ModelArchitectureRegistry()
            models = model_registry.get_all_model_families()
            
            # Format the response
            model_list = []
            for model_family, info in models.items():
                model_list.append({
                    "name": model_family,
                    "family": info.get("family", ""),
                    "type": info.get("type", ""),
                    "modality": info.get("modality", "")
                })
                
            return model_list
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def get_available_hardware(self) -> List[Dict[str, str]]:
        """
        Get a list of available hardware platforms for benchmarking.
        
        Returns:
            List of hardware platform information
        """
        try:
            # Load hardware detection module
            from benchmark_core.hardware import HardwareManager
            
            # Get available hardware
            hardware_manager = HardwareManager()
            available_hardware = hardware_manager.get_available_hardware()
            
            # Format the response
            hardware_list = []
            for hw_type, hw_info in available_hardware.items():
                hardware_list.append({
                    "name": hw_type,
                    "device": hw_info.get("device_name", ""),
                    "memory_gb": hw_info.get("memory_gb", 0),
                    "simulation_mode": hw_info.get("simulation_mode", False)
                })
                
            return hardware_list
            
        except Exception as e:
            logger.error(f"Error getting available hardware: {e}")
            return [
                {"name": "cpu", "device": "CPU", "memory_gb": 0, "simulation_mode": False}
            ]  # Return CPU as fallback
    
    def get_benchmark_reports(self) -> List[Dict[str, str]]:
        """
        Get a list of available benchmark reports.
        
        Returns:
            List of report information
        """
        reports = []
        
        try:
            # Check the results directory for report files
            for run_dir in os.listdir(self.results_dir):
                run_path = os.path.join(self.results_dir, run_dir)
                
                if not os.path.isdir(run_path):
                    continue
                    
                # Look for report files
                report_files = []
                for file in os.listdir(run_path):
                    if file.endswith(".md") or file.endswith(".html"):
                        report_files.append(file)
                        
                if report_files:
                    # Get the run status if available
                    status = "unknown"
                    if run_dir in self.active_runs:
                        status = self.active_runs[run_dir]["status"]
                        
                    reports.append({
                        "run_id": run_dir,
                        "status": status,
                        "reports": report_files,
                        "path": run_path
                    })
            
            # Sort by most recent first (based on run ID if it's a UUID with timestamp)
            reports.sort(key=lambda x: x["run_id"], reverse=True)
            
            return reports
            
        except Exception as e:
            logger.error(f"Error getting benchmark reports: {e}")
            return []
    
    def query_benchmark_results(self, 
                               model: Optional[str] = None, 
                               hardware: Optional[str] = None,
                               batch_size: Optional[int] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query benchmark results from the database.
        
        Args:
            model: Filter by model name
            hardware: Filter by hardware type
            batch_size: Filter by batch size
            limit: Maximum number of results to return
            
        Returns:
            List of benchmark results
        """
        try:
            # Connect to the database
            with BenchmarkDBContext(self.db_path) as db:
                if db.conn is None:
                    return []
                    
                # Get performance metrics
                metrics = db.get_performance_metrics(
                    model_name=model,
                    hardware_type=hardware,
                    batch_size=batch_size,
                    limit=limit
                )
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error querying benchmark results: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources."""
        # Close all WebSocket connections
        for run_id in list(self.ws_connections.keys()):
            for connection in self.ws_connections[run_id]:
                try:
                    # TODO: Replace with task group - asyncio.create_task(connection.close())
                except:
                    pass
            
        self.ws_connections.clear()

# Create the FastAPI application
app = FastAPI(
    title="Benchmark API Server",
    description="API server for running and monitoring benchmarks",
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

# Create the benchmark manager
manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global manager
    
    # Parse arguments (only during direct execution, not when imported)
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Benchmark API Server")
        parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
        parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to benchmark database")
        parser.add_argument("--results-dir", type=str, default="./benchmark_results", help="Directory for benchmark results")
        args = parser.parse_args()
        
        # Create the benchmark manager with arguments
        manager = BenchmarkManager({
            "db_path": args.db_path,
            "results_dir": args.results_dir
        })
    else:
        # Default configuration when imported as a module
        manager = BenchmarkManager()
    
    logger.info("Benchmark API Server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if manager:
        manager.cleanup()
    
    logger.info("Benchmark API Server stopped")

@app.post("/api/benchmark/run", response_model=BenchmarkRunResponse)
async def start_benchmark_run(
    request: BenchmarkRunRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new benchmark run.
    
    Request body parameters:
    - **priority**: Priority level (critical, high, medium, all)
    - **hardware**: List of hardware to benchmark on
    - **models**: Optional list of specific models to benchmark
    - **batch_sizes**: List of batch sizes to benchmark
    - **precision**: Precision to use (fp32, fp16, etc.)
    - **progressive_mode**: Whether to use progressive complexity mode
    - **incremental**: Whether to run only missing or outdated benchmarks
    - **skillset_dir**: Directory containing skillset implementations
    
    Returns the run ID and status.
    """
    return await manager.start_benchmark_run(request, background_tasks)

@app.get("/api/benchmark/status/{run_id}", response_model=BenchmarkStatus)
async def get_benchmark_status(run_id: str):
    """
    Get the status of a benchmark run.
    
    Parameters:
    - **run_id**: The ID of the run to check
    
    Returns the current status of the benchmark run.
    """
    return manager.get_run_status(run_id)

@app.get("/api/benchmark/results/{run_id}")
async def get_benchmark_results(run_id: str):
    """
    Get the results of a completed benchmark run.
    
    Parameters:
    - **run_id**: The ID of the run to get results for
    
    Returns the results of the benchmark run.
    """
    return manager.get_run_results(run_id)

@app.get("/api/benchmark/models")
async def get_models():
    """
    Get a list of available models for benchmarking.
    
    Returns a list of model information.
    """
    return manager.get_available_models()

@app.get("/api/benchmark/hardware")
async def get_hardware():
    """
    Get a list of available hardware platforms for benchmarking.
    
    Returns a list of hardware platform information.
    """
    return manager.get_available_hardware()

@app.get("/api/benchmark/reports")
async def get_reports():
    """
    Get a list of available benchmark reports.
    
    Returns a list of report information.
    """
    return manager.get_benchmark_reports()

@app.get("/api/benchmark/query")
async def query_results(
    model: Optional[str] = None,
    hardware: Optional[str] = None,
    batch_size: Optional[int] = None,
    limit: int = Query(100, gt=0, le=1000)
):
    """
    Query benchmark results from the database.
    
    Parameters:
    - **model**: Filter by model name
    - **hardware**: Filter by hardware type
    - **batch_size**: Filter by batch size
    - **limit**: Maximum number of results to return
    
    Returns a list of benchmark results.
    """
    return manager.query_benchmark_results(model, hardware, batch_size, limit)

@app.websocket("/api/benchmark/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time benchmark updates.
    
    Parameters:
    - **run_id**: The ID of the run to monitor
    
    Returns real-time updates on the benchmark progress.
    """
    await manager.handle_websocket_connection(websocket, run_id)

def main():
    """Main entry point when run directly."""
    import uvicorn
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to benchmark database")
    parser.add_argument("--results-dir", type=str, default="./benchmark_results", help="Directory for benchmark results")
    args = parser.parse_args()
    
    # Start the server
    logger.info(f"Starting Benchmark API Server on {args.host}:{args.port}")
    logger.info(f"Using database: {args.db_path}")
    logger.info(f"Using results directory: {args.results_dir}")
    
    uvicorn.run(
        "benchmark_api_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()