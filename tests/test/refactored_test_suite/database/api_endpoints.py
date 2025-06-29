#!/usr/bin/env python3
"""
Test Suite Database API Endpoints

This module provides FastAPI endpoints for accessing the DuckDB database
storing test run data, history, and performance metrics.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

# Import FastAPI components
try:
    from fastapi import APIRouter, Query, HTTPException, Path, BackgroundTasks
    from pydantic import BaseModel
except ImportError:
    logging.error("Error: FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# Import database integration
try:
    from database.db_integration import TestDatabaseIntegration
except ImportError:
    # Try to add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from database.db_integration import TestDatabaseIntegration
    except ImportError:
        logging.error("Error: TestDatabaseIntegration not found. Make sure db_integration.py is installed.")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API models
class RunHistoryQuery(BaseModel):
    """Request model for querying run history."""
    limit: int = 100
    status: Optional[str] = None
    model_name: Optional[str] = None
    test_type: Optional[str] = None
    batch_id: Optional[str] = None

class BatchHistoryQuery(BaseModel):
    """Request model for querying batch history."""
    limit: int = 100
    status: Optional[str] = None

class PerformanceReportQuery(BaseModel):
    """Request model for querying performance reports."""
    days: int = 30
    model_name: Optional[str] = None
    
class SearchRunsQuery(BaseModel):
    """Request model for searching runs."""
    query: str
    limit: int = 100
    
class ExportDatabaseRequest(BaseModel):
    """Request model for exporting database."""
    filename: str
    include_metrics: bool = True
    include_runs: bool = True
    include_batches: bool = True
    
# Create router
router = APIRouter(
    prefix="/api/test/db",
    tags=["test-db"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

# Database integration instance
db_integration = None

def get_db_integration() -> TestDatabaseIntegration:
    """Get the database integration instance."""
    global db_integration
    if db_integration is None:
        db_integration = TestDatabaseIntegration()
    return db_integration

# API Endpoints
@router.get("/runs")
async def get_run_history(
    limit: int = Query(100, gt=0, le=1000),
    status: Optional[str] = None,
    model_name: Optional[str] = None,
    test_type: Optional[str] = None
):
    """
    Get test run history from the database.
    
    Parameters:
    - **limit**: Maximum number of runs to return
    - **status**: Optional status filter
    - **model_name**: Optional model name filter
    - **test_type**: Optional test type filter
    
    Returns a list of test run history records.
    """
    db = get_db_integration()
    return db.get_run_history(
        limit=limit, 
        model_name=model_name,
        test_type=test_type,
        status=status
    )

@router.get("/batches")
async def get_batch_history(
    limit: int = Query(100, gt=0, le=1000),
    status: Optional[str] = None
):
    """
    Get batch test run history from the database.
    
    Parameters:
    - **limit**: Maximum number of batches to return
    - **status**: Optional status filter
    
    Returns a list of batch history records.
    """
    db = get_db_integration()
    return db.get_batch_history(limit=limit, status=status)

@router.get("/models/stats")
async def get_model_statistics():
    """
    Get statistics for each model.
    
    Returns a list of model statistics.
    """
    db = get_db_integration()
    return db.get_model_statistics()

@router.get("/hardware/stats")
async def get_hardware_statistics():
    """
    Get statistics for each hardware platform.
    
    Returns a list of hardware statistics.
    """
    db = get_db_integration()
    return db.get_hardware_statistics()

@router.get("/run/{run_id}")
async def get_run_details(run_id: str = Path(..., description="The ID of the test run")):
    """
    Get detailed information about a test run.
    
    Parameters:
    - **run_id**: The ID of the test run
    
    Returns detailed test run information.
    """
    db = get_db_integration()
    run = db.get_run_details(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")
    return run

@router.get("/batch/{batch_id}")
async def get_batch_details(batch_id: str = Path(..., description="The ID of the batch")):
    """
    Get detailed information about a batch test run.
    
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
    days: int = Query(30, gt=0, le=365),
    model_name: Optional[str] = None
):
    """
    Get a performance report for the specified time period.
    
    Parameters:
    - **days**: Number of days to include
    - **model_name**: Optional filter by model name
    
    Returns a performance report.
    """
    db = get_db_integration()
    return db.get_performance_report(days=days, model_name=model_name)

@router.post("/search")
async def search_runs(
    request: SearchRunsQuery
):
    """
    Search test runs by query string.
    
    Request body parameters:
    - **query**: Search query string
    - **limit**: Maximum number of results to return
    
    Returns matching test runs.
    """
    db = get_db_integration()
    return db.search_runs(query=request.query, limit=request.limit)

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
    - **include_runs**: Whether to include runs
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

@router.delete("/run/{run_id}")
async def delete_run(run_id: str = Path(..., description="The ID of the test run to delete")):
    """
    Delete a test run from the database.
    
    Parameters:
    - **run_id**: The ID of the test run
    
    Returns status of the delete operation.
    """
    db = get_db_integration()
    
    # Check if the run exists
    run = db.get_run_details(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")
    
    # Delete the run
    success = db.db.delete_test_run(run_id)
    
    return {
        "status": "success" if success else "error",
        "message": f"Test run {run_id} deleted successfully" if success else f"Failed to delete test run {run_id}",
        "timestamp": datetime.datetime.now().isoformat()
    }

@router.delete("/batch/{batch_id}")
async def delete_batch(
    batch_id: str = Path(..., description="The ID of the batch to delete"),
    delete_runs: bool = Query(False, description="Whether to delete all runs in the batch")
):
    """
    Delete a batch from the database.
    
    Parameters:
    - **batch_id**: The ID of the batch
    - **delete_runs**: Whether to delete all runs in the batch
    
    Returns status of the delete operation.
    """
    db = get_db_integration()
    
    # Check if the batch exists
    batch = db.get_batch_details(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    # Delete the batch
    success = db.db.delete_batch_test_run(batch_id, delete_runs)
    
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