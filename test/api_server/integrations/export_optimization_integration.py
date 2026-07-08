#!/usr/bin/env python3
"""
Export Optimization Integration for Unified API Server

This module integrates the Optimization Exporter with the Unified API Server.
It provides REST API endpoints for exporting hardware optimization recommendations
to deployable configuration files.
"""

import os
import sys
import json
import logging
import anyio
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, BinaryIO

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("export_optimization_integration")

# Import export components
try:
    from test.optimization_recommendation.optimization_exporter import OptimizationExporter
    EXPORTER_AVAILABLE = True
except ImportError:
    logger.warning("Optimization Exporter not available")
    EXPORTER_AVAILABLE = False

class ExportOptimizationIntegration:
    """Integration class for Export Optimization with Unified API Server."""
    
    def __init__(
        self,
        benchmark_db_path: str = "benchmark_db.duckdb",
        api_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        output_dir: str = "./optimization_exports",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the export optimization integration.
        
        Args:
            benchmark_db_path: Path to benchmark database
            api_url: API base URL
            api_key: Optional API key
            output_dir: Directory for generated exports
            config: Optional configuration
        """
        self.benchmark_db_path = benchmark_db_path
        self.api_url = api_url
        self.api_key = api_key
        self.output_dir = output_dir
        self.config = config or {}
        
        # Initialize exporter
        self.exporter = None
        if EXPORTER_AVAILABLE:
            try:
                self.exporter = OptimizationExporter(
                    output_dir=output_dir,
                    benchmark_db_path=benchmark_db_path,
                    api_url=api_url,
                    api_key=api_key
                )
                logger.info("Optimization exporter initialized")
            except Exception as e:
                logger.error(f"Error initializing optimization exporter: {e}")
        
        # Task storage
        self.tasks = {}
    
    def register_routes(self, app):
        """
        Register API routes with the FastAPI app.
        
        Args:
            app: FastAPI application
        """
        if not EXPORTER_AVAILABLE:
            logger.warning("Optimization Exporter not available, skipping route registration")
            return
        
        try:
            from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
            from pydantic import BaseModel, Field
            
            # Create router
            router = APIRouter(prefix="/api/export-optimization", tags=["Export Optimization"])
            
            # Define models
            class ExportRequest(BaseModel):
                model_name: str = Field(..., description="Name of the model")
                hardware_platform: str = Field(..., description="Hardware platform")
                recommendation_name: Optional[str] = Field(None, description="Specific recommendation name")
                output_format: str = Field("all", description="Output format (python, json, yaml, script, all)")
                framework: Optional[str] = Field(None, description="Deep learning framework")
            
            class BatchExportRequest(BaseModel):
                recommendations_report: Dict[str, Any] = Field(..., description="Recommendations report")
                output_format: str = Field("all", description="Output format (python, json, yaml, script, all)")
                
            class TaskResponse(BaseModel):
                task_id: str = Field(..., description="Task ID")
                status: str = Field("pending", description="Task status")
                created_at: str = Field(..., description="Task creation timestamp")
                
            # Background task handler
            async def run_in_background(func, task_id, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    self.tasks[task_id] = {
                        "status": "completed",
                        "result": result,
                        "completed_at": datetime.datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Error in background task {task_id}: {e}")
                    self.tasks[task_id] = {
                        "status": "failed",
                        "error": str(e),
                        "completed_at": datetime.datetime.now().isoformat()
                    }
            
            # Routes
            @router.post("/export", response_model=TaskResponse, summary="Export optimization to files")
            async def export_optimization(
                request: ExportRequest,
                background_tasks: BackgroundTasks
            ):
                """
                Export an optimization recommendation to deployable configuration files.
                
                This endpoint generates framework-specific implementation files, configuration
                files, and documentation for a specific optimization recommendation.
                """
                # Create task
                task_id = f"export-{datetime.datetime.now().timestamp()}"
                self.tasks[task_id] = {
                    "status": "pending",
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                # Create custom output directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(
                    self.output_dir,
                    f"{request.model_name}_{request.hardware_platform}_{timestamp}"
                )
                
                # Schedule background task
                background_tasks.add_task(
                    run_in_background,
                    self.exporter.export_optimization,
                    task_id,
                    model_name=request.model_name,
                    hardware_platform=request.hardware_platform,
                    recommendation_name=request.recommendation_name,
                    output_format=request.output_format,
                    output_dir=output_dir
                )
                
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "created_at": self.tasks[task_id]["created_at"]
                }
            
            @router.post("/batch-export", response_model=TaskResponse, summary="Export batch optimizations")
            async def batch_export(
                request: BatchExportRequest,
                background_tasks: BackgroundTasks
            ):
                """
                Export multiple optimization recommendations to deployable configuration files.
                
                This endpoint processes a batch of recommendations from a report and generates
                configuration files and implementation code for each recommendation.
                """
                # Create task
                task_id = f"batch-export-{datetime.datetime.now().timestamp()}"
                self.tasks[task_id] = {
                    "status": "pending",
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                # Create custom output directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(
                    self.output_dir,
                    f"batch_export_{timestamp}"
                )
                
                # Schedule background task
                background_tasks.add_task(
                    run_in_background,
                    self.exporter.export_batch_optimizations,
                    task_id,
                    recommendations_report=request.recommendations_report,
                    output_dir=output_dir,
                    output_format=request.output_format
                )
                
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "created_at": self.tasks[task_id]["created_at"]
                }
            
            @router.get("/task/{task_id}", summary="Get task status and result")
            def get_task_status(task_id: str):
                """
                Get the status and result of a task.
                
                This endpoint returns the current status of a task and its result
                if completed.
                """
                if task_id not in self.tasks:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
                
                return self.tasks[task_id]
            
            @router.get("/download/{task_id}", summary="Download export as ZIP archive")
            async def download_export(task_id: str):
                """
                Download the exported optimization files as a ZIP archive.
                
                This endpoint creates a ZIP archive of all files generated for a specific export task
                and returns it as a file download.
                """
                from fastapi.responses import StreamingResponse
                
                # Check if task exists
                if task_id not in self.tasks:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
                
                # Check if task is completed
                task = self.tasks[task_id]
                if task["status"] != "completed":
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Task {task_id} is not completed (status: {task['status']})"
                    )
                
                # Check if result is available
                if "result" not in task:
                    raise HTTPException(status_code=500, detail=f"No result available for task {task_id}")
                
                # Create archive of exported files
                result = task["result"]
                archive_data = self.exporter.create_archive(result)
                
                if not archive_data:
                    raise HTTPException(status_code=500, detail="Failed to create archive")
                
                # Get filename for archive
                filename = self.exporter.get_archive_filename(result)
                
                # Return streaming response with ZIP data
                return StreamingResponse(
                    iter([archive_data.getvalue()]),
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"'
                    }
                )
            
            @router.get("/templates", summary="List available implementation templates")
            def list_templates():
                """
                List available implementation templates.
                
                This endpoint returns the available implementation templates that
                can be used for exporting optimizations.
                """
                if not self.exporter:
                    raise HTTPException(status_code=500, detail="Exporter not available")
                
                templates = {}
                
                try:
                    # List templates by framework
                    template_dir = self.exporter.template_dir
                    
                    for framework in ["pytorch", "tensorflow", "openvino", "webgpu", "webnn"]:
                        framework_dir = template_dir / framework
                        if framework_dir.exists():
                            templates[framework] = []
                            for template_file in framework_dir.glob("*.py"):
                                template_name = template_file.stem
                                templates[framework].append(template_name)
                except Exception as e:
                    logger.error(f"Error listing templates: {e}")
                
                return {"templates": templates}
            
            # Register router with app
            app.include_router(router)
            logger.info("Export Optimization API routes registered")
            
        except ImportError as e:
            logger.error(f"Error importing FastAPI components: {e}")
        except Exception as e:
            logger.error(f"Error registering routes: {e}")
    
    def close(self):
        """Close connections."""
        if self.exporter:
            self.exporter.close()

# Factory function for integration
def create_integration(
    benchmark_db_path: str = "benchmark_db.duckdb",
    api_url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    output_dir: str = "./optimization_exports",
    config: Optional[Dict[str, Any]] = None
) -> ExportOptimizationIntegration:
    """
    Create an export optimization integration instance.
    
    Args:
        benchmark_db_path: Path to benchmark database
        api_url: API base URL
        api_key: Optional API key
        output_dir: Directory for generated exports
        config: Optional configuration
        
    Returns:
        Export optimization integration instance
    """
    return ExportOptimizationIntegration(
        benchmark_db_path=benchmark_db_path,
        api_url=api_url,
        api_key=api_key,
        output_dir=output_dir,
        config=config
    )