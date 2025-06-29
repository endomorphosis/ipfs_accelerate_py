#!/usr/bin/env python3
"""
Generator Database API Endpoints

This module provides FastAPI endpoints for accessing the DuckDB database
storing generator task data, history, and performance metrics.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FastAPI components
try:
    from fastapi import APIRouter, Query, HTTPException, Path, BackgroundTasks
    from pydantic import BaseModel
except ImportError:
    logging.error("Error: FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# Import database integration
try:
    from database.db_integration import GeneratorDatabaseIntegration
except ImportError:
    logging.error("Error: GeneratorDatabaseIntegration not found. Make sure db_integration.py is installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API models
class TaskHistoryQuery(BaseModel):
    """Request model for querying task history."""
    limit: int = 100
    status: Optional[str] = None
    model_name: Optional[str] = None
    batch_id: Optional[str] = None

class BatchHistoryQuery(BaseModel):
    """Request model for querying batch history."""
    limit: int = 100
    status: Optional[str] = None

class PerformanceReportQuery(BaseModel):
    """Request model for querying performance reports."""
    days: int = 30
    
class ExportDatabaseRequest(BaseModel):
    """Request model for exporting database."""
    filename: str
    include_metrics: bool = True
    include_tasks: bool = True
    include_batches: bool = True
    
# Create router
router = APIRouter(
    prefix="/api/generator/db",
    tags=["generator-db"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

# Database integration instance
db_integration = None

def get_db_integration() -> GeneratorDatabaseIntegration:
    """Get the database integration instance."""
    global db_integration
    if db_integration is None:
        db_integration = GeneratorDatabaseIntegration()
    return db_integration

# API Endpoints
@router.get("/tasks")
async def get_task_history(
    limit: int = Query(100, gt=0, le=1000),
    status: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_id: Optional[str] = None
):
    """
    Get task history from the database.
    
    Parameters:
    - **limit**: Maximum number of tasks to return
    - **status**: Optional status filter
    - **model_name**: Optional model name filter
    - **batch_id**: Optional batch ID filter
    
    Returns a list of task history records.
    """
    db = get_db_integration()
    return db.get_task_history(limit, model_name)

@router.get("/batches")
async def get_batch_history(
    limit: int = Query(100, gt=0, le=1000),
    status: Optional[str] = None
):
    """
    Get batch task history from the database.
    
    Parameters:
    - **limit**: Maximum number of batches to return
    - **status**: Optional status filter
    
    Returns a list of batch history records.
    """
    db = get_db_integration()
    return db.get_batch_history(limit)

@router.get("/models/stats")
async def get_model_statistics():
    """
    Get statistics for each model.
    
    Returns a list of model statistics.
    """
    db = get_db_integration()
    return db.get_model_statistics()

@router.get("/task/{task_id}")
async def get_task_details(task_id: str = Path(..., description="The ID of the task")):
    """
    Get detailed information about a task.
    
    Parameters:
    - **task_id**: The ID of the task
    
    Returns detailed task information.
    """
    db = get_db_integration()
    task = db.get_task_details(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task

@router.get("/batch/{batch_id}")
async def get_batch_details(batch_id: str = Path(..., description="The ID of the batch")):
    """
    Get detailed information about a batch task.
    
    Parameters:
    - **batch_id**: The ID of the batch
    
    Returns detailed batch information.
    """
    db = get_db_integration()
    batch = db.get_batch_details(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return batch

@router.get("/performance")
async def get_performance_report(
    days: int = Query(30, gt=0, le=365)
):
    """
    Get a performance report for the specified time period.
    
    Parameters:
    - **days**: Number of days to include
    
    Returns a performance report.
    """
    db = get_db_integration()
    return db.get_performance_report(days)

@router.post("/export")
async def export_database(
    request: ExportDatabaseRequest,
    background_tasks: BackgroundTasks
):
    """
    Export the database to a JSON file.
    
    Request body parameters:
    - **filename**: Path to the output file
    - **include_metrics**: Whether to include metrics
    - **include_tasks**: Whether to include tasks
    - **include_batches**: Whether to include batches
    
    Returns status of the export operation.
    """
    db = get_db_integration()
    
    # Add task to background tasks
    def do_export():
        try:
            success = db.export_database(request.filename)
            logger.info(f"Database export to {request.filename} {'succeeded' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Error exporting database: {e}")
    
    background_tasks.add_task(do_export)
    
    return {
        "status": "started",
        "message": f"Exporting database to {request.filename}",
        "timestamp": datetime.datetime.now().isoformat()
    }

@router.delete("/task/{task_id}")
async def delete_task(task_id: str = Path(..., description="The ID of the task to delete")):
    """
    Delete a task from the database.
    
    Parameters:
    - **task_id**: The ID of the task
    
    Returns status of the delete operation.
    """
    db = get_db_integration()
    
    # Check if the task exists
    task = db.get_task_details(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Delete the task
    success = db.db.delete_task(task_id)
    
    return {
        "status": "success" if success else "error",
        "message": f"Task {task_id} deleted successfully" if success else f"Failed to delete task {task_id}",
        "timestamp": datetime.datetime.now().isoformat()
    }

@router.delete("/batch/{batch_id}")
async def delete_batch(
    batch_id: str = Path(..., description="The ID of the batch to delete"),
    delete_tasks: bool = Query(False, description="Whether to delete all tasks in the batch")
):
    """
    Delete a batch from the database.
    
    Parameters:
    - **batch_id**: The ID of the batch
    - **delete_tasks**: Whether to delete all tasks in the batch
    
    Returns status of the delete operation.
    """
    db = get_db_integration()
    
    # Check if the batch exists
    batch = db.get_batch_details(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    # Delete the batch
    success = db.db.delete_batch_task(batch_id, delete_tasks)
    
    return {
        "status": "success" if success else "error",
        "message": f"Batch {batch_id} deleted successfully" if success else f"Failed to delete batch {batch_id}",
        "timestamp": datetime.datetime.now().isoformat()
    }
    
def init_api(app=None):
    """Initialize the API and register routes with the app."""
    if app:
        app.include_router(router)
    return router