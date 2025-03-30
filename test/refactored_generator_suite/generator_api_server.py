#!/usr/bin/env python3
"""
Generator API Server

This module provides a FastAPI server for generating model implementations,
allowing for code generation, template management, and hardware-specific
optimizations through RESTful APIs and WebSockets.
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
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

# Import database integration
try:
    from refactored_generator_suite.database.db_integration import GeneratorDatabaseIntegration
    from refactored_generator_suite.database.api_endpoints import init_api as init_db_api
    DATABASE_INTEGRATION_AVAILABLE = True
except ImportError:
    print("Warning: Database integration not available. Using memory-only storage.")
    DATABASE_INTEGRATION_AVAILABLE = False

# Import generator components
try:
    from refactored_generator_suite.generator_core.config import ConfigManager
    from refactored_generator_suite.generator_core.registry import ComponentRegistry
    from refactored_generator_suite.generator_core.generator import GeneratorCore
    from refactored_generator_suite.hardware.hardware_detection import HardwareManager
    GENERATOR_COMPONENTS_AVAILABLE = True
except ImportError:
    print("Warning: Generator core components not available. Using mock implementations.")
    GENERATOR_COMPONENTS_AVAILABLE = False
    # Define mock classes for testing
    class ConfigManager:
        def __init__(self): self.config = {}
        def get(self, key, default=None): return self.config.get(key, default)
    class ComponentRegistry:
        def get_template(self, model_type): return MockTemplate()
        def get_model_info(self, model_type): return {"architecture": "mock", "name": model_type}
        def get_hardware_info(self): return {"cpu": {"available": True}}
    class GeneratorCore:
        def __init__(self, *args, **kwargs): pass
        def generate(self, model_type, options):
            time.sleep(1)  # Simulate work
            return {
                "success": True,
                "output_file": f"./mock_output/test_{model_type}.py",
                "model_type": model_type,
                "model_info": {"architecture": "mock", "name": model_type},
                "architecture": "mock",
                "duration": 1.0
            }
    class HardwareManager:
        def detect_all(self):
            return {"cpu": {"available": True}}
    class MockTemplate:
        def render(self, context):
            return f"# Mock implementation for {context['model_type']}\n\ndef test_mock():\n    assert True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("generator_api_server.log")]
)
logger = logging.getLogger("generator_api_server")

# Define API models
class GenerateModelRequest(BaseModel):
    """Request model for generating a model implementation."""
    model_name: str
    hardware: List[str] = ["cpu"]
    output_dir: Optional[str] = None
    force: bool = False
    template_type: Optional[str] = None
    template_context: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    fix_syntax: bool = True

class GenerateModelResponse(BaseModel):
    """Response model for model generation requests."""
    task_id: str
    status: str
    message: str
    started_at: str

class GenerateModelStatusResponse(BaseModel):
    """Status response model for a model generation task."""
    task_id: str
    status: str
    progress: float
    model_name: str
    current_step: str
    started_at: str
    elapsed_time: float
    estimated_remaining_time: Optional[float] = None
    output_file: Optional[str] = None
    error: Optional[str] = None

class GenerateModelResultResponse(BaseModel):
    """Result response model for a completed generation task."""
    task_id: str
    status: str
    model_name: str
    output_file: Optional[str] = None
    architecture: Optional[str] = None
    template_type: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    duration: float
    error: Optional[str] = None

class ModelTemplateInfo(BaseModel):
    """Information about a model template."""
    name: str
    type: str
    description: str
    supported_hardware: List[str]

class HardwareSupportInfo(BaseModel):
    """Information about hardware support."""
    name: str
    available: bool
    details: Optional[Dict[str, Any]] = None

class GeneratorTaskManager:
    """
    Manager class for handling generator operations.
    
    This class provides methods for generating model implementations,
    checking status, and handling WebSocket connections for real-time updates.
    """
    
    def __init__(self, config_file: Optional[str] = None, db_path: Optional[str] = None):
        """Initialize the generator task manager.
        
        Args:
            config_file: Optional path to a config file
            db_path: Optional path to database file
        """
        self.active_tasks = {}
        self.ws_connections = {}
        
        # Initialize generator components
        if GENERATOR_COMPONENTS_AVAILABLE:
            self.config = ConfigManager(config_file)
            self.registry = ComponentRegistry()
            self.hardware_manager = HardwareManager()
            self.generator = GeneratorCore(self.config, self.registry, self.hardware_manager)
            
            # Get output directory from config or use default
            self.output_dir = Path(self.config.get("output_dir", "./generated_models"))
        else:
            self.config = ConfigManager()
            self.registry = ComponentRegistry()
            self.hardware_manager = HardwareManager()
            self.generator = GeneratorCore(self.config, self.registry, self.hardware_manager)
            self.output_dir = Path("./generated_models")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize database integration if available
        self.db_integration = None
        if DATABASE_INTEGRATION_AVAILABLE:
            try:
                self.db_integration = GeneratorDatabaseIntegration(db_path)
                logger.info(f"Database integration initialized")
            except Exception as e:
                logger.error(f"Error initializing database integration: {e}")
                self.db_integration = None
    
    async def generate_model(self, 
                           request: GenerateModelRequest, 
                           background_tasks: BackgroundTasks) -> GenerateModelResponse:
        """
        Generate a model implementation asynchronously.
        
        Args:
            request: Model generation request parameters
            background_tasks: FastAPI background tasks object
            
        Returns:
            Response with task ID and status
        """
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Set output directory
        output_dir = request.output_dir or str(self.output_dir)
        
        # Create task configuration
        task_data = {
            "task_id": task_id,
            "status": "initializing",
            "progress": 0.0,
            "current_step": "Setting up generation environment",
            "started_at": datetime.now(),
            "completed_at": None,
            "model_name": request.model_name,
            "hardware": request.hardware,
            "output_dir": output_dir,
            "template_type": request.template_type,
            "force": request.force,
            "output_file": None,
            "error": None,
            "template_context": request.template_context
        }
        
        # Store in memory
        self.active_tasks[task_id] = task_data
        
        # Store in database if available
        if self.db_integration:
            try:
                self.db_integration.track_task_start(task_data)
            except Exception as e:
                logger.error(f"Error storing task in database: {e}")
        
        # Log start of task
        logger.info(f"Starting model generation task {task_id} for model {request.model_name}")
        
        # Add the task to run asynchronously
        background_tasks.add_task(
            self._generate_model,
            task_id=task_id,
            model_name=request.model_name,
            hardware=request.hardware,
            output_dir=output_dir,
            template_type=request.template_type,
            template_context=request.template_context,
            task=request.task,
            force=request.force,
            fix_syntax=request.fix_syntax
        )
        
        # Return the response
        return GenerateModelResponse(
            task_id=task_id,
            status="initializing",
            message=f"Model generation started for {request.model_name}",
            started_at=self.active_tasks[task_id]["started_at"].isoformat()
        )
    
    async def _generate_model(self,
                            task_id: str,
                            model_name: str,
                            hardware: List[str],
                            output_dir: str,
                            template_type: Optional[str] = None,
                            template_context: Optional[Dict[str, Any]] = None,
                            task: Optional[str] = None,
                            force: bool = False,
                            fix_syntax: bool = True):
        """
        Generate the model implementation in the background.
        
        Args:
            task_id: Unique ID for this task
            model_name: Name of the model to generate
            hardware: List of hardware to support
            output_dir: Output directory for the generated file
            template_type: Type of template to use
            template_context: Additional context for the template
            task: Task to target (text, vision, audio, etc.)
            force: Whether to force overwrite existing files
            fix_syntax: Whether to fix syntax errors in generated code
        """
        try:
            # Update status
            self.active_tasks[task_id]["status"] = "running"
            self.active_tasks[task_id]["current_step"] = "Preparing generation environment"
            self.active_tasks[task_id]["progress"] = 0.1
            
            # Update database
            if self.db_integration:
                try:
                    self.db_integration.track_task_update(
                        task_id, 
                        "running", 
                        0.1, 
                        "Preparing generation environment"
                    )
                except Exception as e:
                    logger.error(f"Error updating task in database: {e}")
            
            await self._send_ws_update(task_id)
            
            # Determine output filename
            self.active_tasks[task_id]["current_step"] = "Determining output filename"
            self.active_tasks[task_id]["progress"] = 0.15
            
            # Update database
            if self.db_integration:
                try:
                    self.db_integration.track_task_update(
                        task_id, 
                        "running", 
                        0.15, 
                        "Determining output filename"
                    )
                except Exception as e:
                    logger.error(f"Error updating task in database: {e}")
            
            await self._send_ws_update(task_id)
            
            # Default filename for the model
            filename = f"test_{model_name.replace('-', '_')}.py"
            output_file = os.path.join(output_dir, filename)
            
            # Check if the file already exists
            if os.path.exists(output_file) and not force:
                self.active_tasks[task_id]["status"] = "error"
                self.active_tasks[task_id]["error"] = f"Output file {output_file} already exists. Use 'force=True' to overwrite."
                self.active_tasks[task_id]["current_step"] = "Error: File already exists"
                self.active_tasks[task_id]["completed_at"] = datetime.now()
                
                # Update database
                if self.db_integration:
                    try:
                        self.db_integration.track_task_update(
                            task_id, 
                            "error", 
                            0.0, 
                            "Error: File already exists", 
                            f"Output file {output_file} already exists. Use 'force=True' to overwrite."
                        )
                    except Exception as e:
                        logger.error(f"Error updating task in database: {e}")
                
                await self._send_ws_update(task_id)
                return
            
            # Prepare options for generator
            options = {
                "output_dir": output_dir,
                "output_file": output_file,
                "hardware": hardware,
                "fix_syntax": fix_syntax,
                "template_type": template_type,
                "task": task
            }
            
            if template_context:
                options["template_context"] = template_context
            
            # Update status
            self.active_tasks[task_id]["progress"] = 0.3
            self.active_tasks[task_id]["current_step"] = f"Generating implementation for {model_name}"
            
            # Update database
            if self.db_integration:
                try:
                    self.db_integration.track_task_update(
                        task_id, 
                        "running", 
                        0.3, 
                        f"Generating implementation for {model_name}"
                    )
                except Exception as e:
                    logger.error(f"Error updating task in database: {e}")
            
            await self._send_ws_update(task_id)
            
            # Generate the model implementation
            result = await asyncio.to_thread(self.generator.generate, model_name, options)
            
            # Update the task with the result
            if result["success"]:
                self.active_tasks[task_id]["status"] = "completed"
                self.active_tasks[task_id]["progress"] = 1.0
                self.active_tasks[task_id]["current_step"] = "Generation completed"
                self.active_tasks[task_id]["output_file"] = result["output_file"]
                
                if "architecture" in result:
                    self.active_tasks[task_id]["architecture"] = result["architecture"]
                    
                self.active_tasks[task_id]["completed_at"] = datetime.now()
                
                # Update database
                if self.db_integration:
                    try:
                        self.db_integration.track_task_completion(task_id, result)
                    except Exception as e:
                        logger.error(f"Error updating completion in database: {e}")
            else:
                self.active_tasks[task_id]["status"] = "error"
                self.active_tasks[task_id]["error"] = result.get("error", "Unknown error")
                self.active_tasks[task_id]["current_step"] = "Error during generation"
                self.active_tasks[task_id]["completed_at"] = datetime.now()
                
                # Update database
                if self.db_integration:
                    try:
                        self.db_integration.track_task_update(
                            task_id, 
                            "error", 
                            1.0, 
                            "Error during generation", 
                            result.get("error", "Unknown error")
                        )
                    except Exception as e:
                        logger.error(f"Error updating error in database: {e}")
            
            await self._send_ws_update(task_id)
            logger.info(f"Model generation task {task_id} completed with status: {self.active_tasks[task_id]['status']}")
            
        except Exception as e:
            logger.error(f"Error generating model {model_name} in task {task_id}: {e}", exc_info=True)
            self.active_tasks[task_id]["status"] = "error"
            self.active_tasks[task_id]["error"] = str(e)
            self.active_tasks[task_id]["current_step"] = "Error during generation"
            self.active_tasks[task_id]["completed_at"] = datetime.now()
            
            # Update database
            if self.db_integration:
                try:
                    self.db_integration.track_task_update(
                        task_id, 
                        "error", 
                        1.0, 
                        "Error during generation", 
                        str(e)
                    )
                except Exception as db_error:
                    logger.error(f"Error updating exception in database: {db_error}")
            
            await self._send_ws_update(task_id)
    
    async def _send_ws_update(self, task_id: str):
        """
        Send an update to all WebSocket connections for a task.
        
        Args:
            task_id: The ID of the task that was updated
        """
        if task_id not in self.ws_connections:
            return
            
        # Get the current status
        status_data = self.get_task_status(task_id)
        
        # Send to all connected clients
        for connection in self.ws_connections[task_id]:
            try:
                await connection.send_json(status_data)
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {e}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a generation task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Dict containing task status information
            
        Raises:
            HTTPException: If the task doesn't exist
        """
        if task_id not in self.active_tasks:
            raise HTTPException(status_code=404, detail=f"Generation task {task_id} not found")
            
        task_data = self.active_tasks[task_id]
        
        # Calculate elapsed time
        elapsed = (datetime.now() - task_data["started_at"]).total_seconds()
        
        # Calculate estimated remaining time
        remaining = None
        if task_data["progress"] > 0 and task_data["progress"] < 1.0:
            remaining = (elapsed / task_data["progress"]) * (1.0 - task_data["progress"])
        
        # Create status object
        status = {
            "task_id": task_id,
            "status": task_data["status"],
            "progress": task_data["progress"],
            "model_name": task_data["model_name"],
            "current_step": task_data["current_step"],
            "started_at": task_data["started_at"].isoformat(),
            "elapsed_time": elapsed,
            "estimated_remaining_time": remaining
        }
        
        # Include output file if available
        if task_data.get("output_file"):
            status["output_file"] = task_data["output_file"]
            
        # Include error if available
        if task_data.get("error"):
            status["error"] = task_data["error"]
            
        return status
    
    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed generation task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Dict containing generation result
            
        Raises:
            HTTPException: If the task doesn't exist or isn't completed
        """
        if task_id not in self.active_tasks:
            raise HTTPException(status_code=404, detail=f"Generation task {task_id} not found")
            
        task_data = self.active_tasks[task_id]
        
        if task_data["status"] not in ["completed", "error"]:
            raise HTTPException(status_code=400, detail=f"Generation task {task_id} is not completed")
            
        # Create result object
        result = {
            "task_id": task_id,
            "status": task_data["status"],
            "model_name": task_data["model_name"],
            "started_at": task_data["started_at"].isoformat(),
            "completed_at": task_data["completed_at"].isoformat() if task_data.get("completed_at") else None,
            "duration": (task_data.get("completed_at", datetime.now()) - task_data["started_at"]).total_seconds(),
            "template_type": task_data.get("template_type")
        }
        
        # Include output file if available
        if task_data.get("output_file"):
            result["output_file"] = task_data["output_file"]
            
        # Include architecture if available
        if task_data.get("architecture"):
            result["architecture"] = task_data["architecture"]
            
        # Include error if available
        if task_data.get("error"):
            result["error"] = task_data["error"]
            
        return result
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """
        Get a list of available templates.
        
        Returns:
            List of template information
        """
        try:
            # Try to get templates from registry
            if hasattr(self.registry, "get_templates"):
                return self.registry.get_templates()
            
            # Mock implementation
            return [
                {
                    "name": "bert",
                    "type": "encoder",
                    "description": "Template for BERT-like models",
                    "supported_hardware": ["cpu", "cuda", "rocm", "openvino"]
                },
                {
                    "name": "gpt2",
                    "type": "decoder",
                    "description": "Template for GPT-2 and similar models",
                    "supported_hardware": ["cpu", "cuda", "rocm"]
                },
                {
                    "name": "t5",
                    "type": "encoder-decoder",
                    "description": "Template for T5 and similar models",
                    "supported_hardware": ["cpu", "cuda"]
                },
                {
                    "name": "vit",
                    "type": "vision",
                    "description": "Template for Vision Transformer models",
                    "supported_hardware": ["cpu", "cuda", "webgpu"]
                },
                {
                    "name": "whisper",
                    "type": "audio",
                    "description": "Template for Whisper and similar models",
                    "supported_hardware": ["cpu", "cuda"]
                }
            ]
        except Exception as e:
            logger.error(f"Error getting available templates: {e}")
            return []
    
    def get_available_hardware(self) -> List[Dict[str, Any]]:
        """
        Get a list of available hardware platforms.
        
        Returns:
            List of hardware platform information
        """
        try:
            # Get hardware info from hardware manager
            hardware_info = self.hardware_manager.detect_all()
            
            # Format the response
            return [
                {
                    "name": hw_type,
                    "available": info.get("available", False),
                    "details": info
                }
                for hw_type, info in hardware_info.items()
            ]
        except Exception as e:
            logger.error(f"Error getting available hardware: {e}")
            return [
                {"name": "cpu", "available": True, "details": {"cores": 1}}
            ]  # Return CPU as fallback
    
    def get_task_file_content(self, task_id: str) -> str:
        """
        Get the content of the generated file.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Content of the generated file
            
        Raises:
            HTTPException: If the task doesn't exist, isn't completed, or the file doesn't exist
        """
        if task_id not in self.active_tasks:
            raise HTTPException(status_code=404, detail=f"Generation task {task_id} not found")
            
        task_data = self.active_tasks[task_id]
        
        if task_data["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Generation task {task_id} is not completed")
            
        if not task_data.get("output_file"):
            raise HTTPException(status_code=400, detail=f"No output file for task {task_id}")
            
        try:
            with open(task_data["output_file"], 'r') as f:
                return f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    async def handle_websocket_connection(self, websocket: WebSocket, task_id: str):
        """
        Handle a WebSocket connection for a generation task.
        
        Args:
            websocket: The WebSocket connection
            task_id: The ID of the task to monitor
        """
        await websocket.accept()
        
        # Check if the task exists
        if task_id not in self.active_tasks:
            await websocket.send_json({
                "error": f"Generation task {task_id} not found"
            })
            await websocket.close()
            return
            
        # Add the connection to the list for this task
        if task_id not in self.ws_connections:
            self.ws_connections[task_id] = []
            
        self.ws_connections[task_id].append(websocket)
        
        try:
            # Send initial status
            status = self.get_task_status(task_id)
            await websocket.send_json(status)
            
            # Keep the connection open and handle messages
            while True:
                message = await websocket.receive_text()
                
                # Handle client messages if needed
                if message == "ping":
                    await websocket.send_json({"pong": True})
                elif message == "status":
                    status = self.get_task_status(task_id)
                    await websocket.send_json(status)
                    
        except WebSocketDisconnect:
            # Remove the connection from the list
            if task_id in self.ws_connections:
                self.ws_connections[task_id].remove(websocket)
                
                # Clean up empty lists
                if not self.ws_connections[task_id]:
                    del self.ws_connections[task_id]
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            
            # Ensure connection is removed on error
            if task_id in self.ws_connections and websocket in self.ws_connections[task_id]:
                self.ws_connections[task_id].remove(websocket)
                
                # Clean up empty lists
                if not self.ws_connections[task_id]:
                    del self.ws_connections[task_id]
    
    async def batch_generate_models(self, 
                                  model_names: List[str], 
                                  common_options: Dict[str, Any],
                                  background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """
        Generate multiple models in a batch.
        
        Args:
            model_names: List of model names to generate
            common_options: Common options for all models
            background_tasks: FastAPI background tasks object
            
        Returns:
            Dict with batch task information
        """
        # Generate a unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Create task IDs for each model
        task_ids = []
        
        # Create batch task data
        batch_data = {
            "batch_id": batch_id,
            "description": f"Batch generation of {len(model_names)} models",
            "task_count": len(model_names),
            "started_at": datetime.now(),
            "status": "running"
        }
        
        # Store batch in database if available
        if self.db_integration:
            try:
                self.db_integration.db.store_batch_task(batch_data)
            except Exception as e:
                logger.error(f"Error storing batch in database: {e}")
        
        for model_name in model_names:
            # Create a request for this model
            request = GenerateModelRequest(
                model_name=model_name,
                hardware=common_options.get("hardware", ["cpu"]),
                output_dir=common_options.get("output_dir"),
                force=common_options.get("force", False),
                template_type=common_options.get("template_type"),
                template_context=common_options.get("template_context"),
                task=common_options.get("task"),
                fix_syntax=common_options.get("fix_syntax", True)
            )
            
            # Start the generation task
            response = await self.generate_model(request, background_tasks)
            task_ids.append(response.task_id)
            
            # Update task with batch ID in memory
            if response.task_id in self.active_tasks:
                self.active_tasks[response.task_id]["batch_id"] = batch_id
                
                # Update in database if available
                if self.db_integration:
                    try:
                        task_data = self.active_tasks[response.task_id].copy()
                        task_data["batch_id"] = batch_id
                        self.db_integration.db.store_task(task_data)
                    except Exception as e:
                        logger.error(f"Error updating task with batch ID in database: {e}")
        
        # Track batch start in database if available
        if self.db_integration:
            try:
                self.db_integration.track_batch_start(batch_id, model_names, task_ids)
            except Exception as e:
                logger.error(f"Error tracking batch start in database: {e}")
        
        # Return batch information
        return {
            "batch_id": batch_id,
            "task_ids": task_ids,
            "model_count": len(model_names),
            "started_at": datetime.now().isoformat(),
            "status": "running"
        }
    
    def cleanup(self):
        """Clean up resources."""
        # Close all WebSocket connections
        for task_id in list(self.ws_connections.keys()):
            for connection in self.ws_connections[task_id]:
                try:
                    asyncio.create_task(connection.close())
                except:
                    pass
            
        self.ws_connections.clear()
        
        # Close database connection if available
        if self.db_integration:
            try:
                self.db_integration.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

# Create the FastAPI application
app = FastAPI(
    title="Generator API Server",
    description="API server for generating model implementations",
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

# Create the generator manager
manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global manager
    
    # Parse arguments (only during direct execution, not when imported)
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Generator API Server")
        parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
        parser.add_argument("--config", type=str, help="Path to config file")
        parser.add_argument("--output-dir", type=str, default="./generated_models", help="Output directory for generated files")
        parser.add_argument("--db-path", type=str, help="Path to database file")
        args = parser.parse_args()
        
        # Create the generator manager with configuration
        manager = GeneratorTaskManager(args.config, args.db_path)
        
        # Set output directory
        manager.output_dir = Path(args.output_dir)
        os.makedirs(manager.output_dir, exist_ok=True)
    else:
        # Default configuration when imported as a module
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "data", "generator_tasks.duckdb")
        manager = GeneratorTaskManager(db_path=db_path)
    
    # Initialize database API endpoints if available
    if DATABASE_INTEGRATION_AVAILABLE:
        try:
            init_db_api(app)
            logger.info("Database API endpoints initialized")
        except Exception as e:
            logger.error(f"Error initializing database API endpoints: {e}")
    
    logger.info("Generator API Server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if manager:
        manager.cleanup()
    
    logger.info("Generator API Server stopped")

@app.post("/api/generator/model", response_model=GenerateModelResponse)
async def generate_model(
    request: GenerateModelRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a model implementation.
    
    Request body parameters:
    - **model_name**: Name of the model to generate
    - **hardware**: List of hardware platforms to support
    - **output_dir**: Directory to store generated files
    - **force**: Whether to overwrite existing files
    - **template_type**: Type of template to use
    - **template_context**: Additional context for the template
    - **task**: Task to target (text, vision, audio, etc.)
    - **fix_syntax**: Whether to fix syntax errors in generated code
    
    Returns the task ID and status.
    """
    return await manager.generate_model(request, background_tasks)

@app.post("/api/generator/batch")
async def batch_generate(
    model_names: List[str],
    hardware: List[str] = ["cpu"],
    output_dir: Optional[str] = None,
    force: bool = False,
    template_type: Optional[str] = None,
    task: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Generate multiple model implementations in a batch.
    
    Request body parameters:
    - **model_names**: List of model names to generate
    - **hardware**: List of hardware platforms to support
    - **output_dir**: Directory to store generated files
    - **force**: Whether to overwrite existing files
    - **template_type**: Type of template to use
    - **task**: Task to target (text, vision, audio, etc.)
    
    Returns the batch ID, task IDs and status.
    """
    common_options = {
        "hardware": hardware,
        "output_dir": output_dir,
        "force": force,
        "template_type": template_type,
        "task": task
    }
    
    return await manager.batch_generate_models(model_names, common_options, background_tasks)

@app.get("/api/generator/status/{task_id}", response_model=GenerateModelStatusResponse)
async def get_generation_status(task_id: str):
    """
    Get the status of a model generation task.
    
    Parameters:
    - **task_id**: The ID of the task to check
    
    Returns the current status of the generation task.
    """
    return manager.get_task_status(task_id)

@app.get("/api/generator/result/{task_id}", response_model=GenerateModelResultResponse)
async def get_generation_result(task_id: str):
    """
    Get the result of a completed generation task.
    
    Parameters:
    - **task_id**: The ID of the task to get results for
    
    Returns the result of the generation task.
    """
    return manager.get_task_result(task_id)

@app.get("/api/generator/file/{task_id}")
async def get_generated_file(task_id: str):
    """
    Get the content of the generated file.
    
    Parameters:
    - **task_id**: The ID of the task
    
    Returns the content of the generated file.
    """
    content = manager.get_task_file_content(task_id)
    return {"content": content}

@app.get("/api/generator/templates")
async def get_templates():
    """
    Get a list of available templates.
    
    Returns a list of template information.
    """
    return manager.get_available_templates()

@app.get("/api/generator/hardware")
async def get_hardware():
    """
    Get a list of available hardware platforms.
    
    Returns a list of hardware platform information.
    """
    return manager.get_available_hardware()

@app.websocket("/api/generator/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time generation updates.
    
    Parameters:
    - **task_id**: The ID of the task to monitor
    
    Returns real-time updates on the generation progress.
    """
    await manager.handle_websocket_connection(websocket, task_id)

def main():
    """Main entry point when run directly."""
    import uvicorn
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generator API Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./generated_models", help="Output directory for generated files")
    parser.add_argument("--db-path", type=str, default="./data/generator_tasks.duckdb", help="Path to database file")
    args = parser.parse_args()
    
    # Start the server
    logger.info(f"Starting Generator API Server on {args.host}:{args.port}")
    logger.info(f"Using output directory: {args.output_dir}")
    logger.info(f"Using database path: {args.db_path}")
    
    uvicorn.run(
        "generator_api_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()