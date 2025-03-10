#!/usr/bin/env python
"""
Benchmark Database API Server

This script provides a REST API for accessing the benchmark database,
allowing programmatic access to results and supporting integration with
other tools and systems.
"""

import os
import sys
import json
import argparse
import logging
import datetime
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uvicorn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# No need to add parent directory in new structure
# All database modules are now in the duckdb_api directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_api")

# Create the FastAPI app
app = FastAPI(
    title="Benchmark Database API",
    description="API for accessing hardware and model benchmark data",
    version="1.0.0"
)

# Create a directory for templates
templates_dir = Path(__file__).parent / "templates"
os.makedirs(templates_dir, exist_ok=True)

# Create a simple dashboard template
with open(templates_dir / "dashboard.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .chart-container {
            height: 400px;
            margin-bottom: 20px;
        }
        .card {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1>Hardware Benchmark Dashboard</h1>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Hardware Compatibility Matrix</h5>
                    </div>
                    <div class="card-body">
                        <img src="/api/charts/compatibility-matrix" class="img-fluid">
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Performance Overview</h5>
                    </div>
                    <div class="card-body">
                        <img src="/api/charts/performance-overview" class="img-fluid">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Model Family Performance by Hardware</h5>
                    </div>
                    <div class="card-body">
                        <img src="/api/charts/hardware-comparison" class="img-fluid">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Web Platform Audio Model Comparison</h5>
                    </div>
                    <div class="card-body">
                        <img src="/api/charts/web-platform-comparison" class="img-fluid">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Tests</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Test Name</th>
                                    <th>Type</th>
                                    <th>Status</th>
                                    <th>Date</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for test in recent_tests %}
                                <tr>
                                    <td>{{ test.test_name }}</td>
                                    <td>{{ test.test_type }}</td>
                                    <td>
                                        {% if test.success %}
                                        <span class="badge bg-success">Success</span>
                                        {% else %}
                                        <span class="badge bg-danger">Failed</span>
                                        {% endif %}
                                    </td>
                                    <td>{{ test.started_at }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """)

templates = Jinja2Templates(directory=templates_dir)

# Database connection
db_path = None
conn = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run the benchmark database API server")
    
    parser.add_argument("--db", type=str, default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (for development)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--serve", action="store_true",
                        help="Start the API server")
    
    return parser.parse_args()

def get_db():
    """Get a connection to the database"""
    global conn, db_path
    if conn is None:
        if not os.path.exists(db_path):
            raise HTTPException(status_code=500, detail=f"Database file not found: {db_path}")
        
        try:
            conn = duckdb.connect(db_path)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to database: {str(e)}")
    
    return conn

def execute_query(query, params=None):
    """Execute a query with the database connection"""
    try:
        db = get_db()
        if params:
            return db.execute(query, params).fetchdf()
        else:
            return db.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        logger.error(f"Query: {query}")
        if params:
            logger.error(f"Params: {params}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

# API Models
class ModelInfo(BaseModel):
    model_id: int
    model_name: str
    model_family: Optional[str] = None
    modality: Optional[str] = None
    source: Optional[str] = None
    version: Optional[str] = None
    parameters_million: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime

class HardwareInfo(BaseModel):
    hardware_id: int
    hardware_type: str
    device_name: Optional[str] = None
    platform: Optional[str] = None
    platform_version: Optional[str] = None
    driver_version: Optional[str] = None
    memory_gb: Optional[float] = None
    compute_units: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime

class PerformanceResult(BaseModel):
    result_id: int
    model_name: str
    model_family: Optional[str] = None
    hardware_type: str
    device_name: Optional[str] = None
    test_case: str
    batch_size: int
    precision: Optional[str] = None
    total_time_seconds: Optional[float] = None
    average_latency_ms: Optional[float] = None
    throughput_items_per_second: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    created_at: datetime.datetime

class CompatibilityResult(BaseModel):
    compatibility_id: int
    model_name: str
    model_family: Optional[str] = None
    hardware_type: str
    device_name: Optional[str] = None
    is_compatible: bool
    compatibility_score: float
    error_message: Optional[str] = None
    created_at: datetime.datetime

class TestResult(BaseModel):
    test_result_id: int
    test_module: str
    test_class: Optional[str] = None
    test_name: str
    status: str
    model_name: Optional[str] = None
    hardware_type: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime.datetime
    
class WebPlatformResult(BaseModel):
    result_id: int
    model_name: str
    model_type: str
    browser: str
    platform: str
    status: str
    execution_time: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    source_file: Optional[str] = None
    timestamp: datetime.datetime

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Benchmark Database API</title>
        </head>
        <body>
            <h1>Benchmark Database API</h1>
            <p>API for accessing hardware and model benchmark data</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/dashboard">Dashboard</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    # Get recent test runs
    recent_tests = execute_query("""
    SELECT test_name, test_type, success, started_at 
    FROM test_runs 
    ORDER BY started_at DESC 
    LIMIT 10
    """)
    
    # Convert to list of dicts
    recent_tests_list = [
        {
            "test_name": row['test_name'],
            "test_type": row['test_type'],
            "success": row['success'],
            "started_at": row['started_at'].strftime("%Y-%m-%d %H:%M:%S")
        }
        for _, row in recent_tests.iterrows()
    ]
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {"request": {}, "recent_tests": recent_tests_list}
    )

@app.get("/api/models", response_model=List[ModelInfo])
async def get_models(
    family: Optional[str] = None,
    modality: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100
):
    """Get list of models in the database"""
    # Build query with filters
    query = "SELECT * FROM models"
    params = []
    where_clauses = []
    
    if family:
        where_clauses.append("model_family = ?")
        params.append(family)
    
    if modality:
        where_clauses.append("modality = ?")
        params.append(modality)
    
    if source:
        where_clauses.append("source = ?")
        params.append(source)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY model_name LIMIT ?"
    params.append(limit)
    
    # Execute query
    models_df = execute_query(query, params)
    
    # Parse JSON metadata column
    for i, row in models_df.iterrows():
        if pd.notna(row.get('metadata')):
            try:
                models_df.at[i, 'metadata'] = json.loads(row['metadata'])
            except:
                models_df.at[i, 'metadata'] = {}
    
    # Convert to list of dicts
    models_list = models_df.to_dict(orient='records')
    return models_list

@app.get("/api/hardware", response_model=List[HardwareInfo])
async def get_hardware(
    hardware_type: Optional[str] = None,
    platform: Optional[str] = None,
    limit: int = 100
):
    """Get list of hardware platforms in the database"""
    # Build query with filters
    query = "SELECT * FROM hardware_platforms"
    params = []
    where_clauses = []
    
    if hardware_type:
        where_clauses.append("hardware_type = ?")
        params.append(hardware_type)
    
    if platform:
        where_clauses.append("platform = ?")
        params.append(platform)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY hardware_type, device_name LIMIT ?"
    params.append(limit)
    
    # Execute query
    hardware_df = execute_query(query, params)
    
    # Parse JSON metadata column
    for i, row in hardware_df.iterrows():
        if pd.notna(row.get('metadata')):
            try:
                hardware_df.at[i, 'metadata'] = json.loads(row['metadata'])
            except:
                hardware_df.at[i, 'metadata'] = {}
    
    # Convert to list of dicts
    hardware_list = hardware_df.to_dict(orient='records')
    return hardware_list

@app.get("/api/performance", response_model=List[PerformanceResult])
async def get_performance(
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
    hardware_type: Optional[str] = None,
    test_case: Optional[str] = None,
    batch_size: Optional[int] = None,
    precision: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100
):
    """Get performance benchmark results"""
    # Build query with filters
    query = """
    SELECT 
        pr.result_id,
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        pr.test_case,
        pr.batch_size,
        pr.precision,
        pr.total_time_seconds,
        pr.average_latency_ms,
        pr.throughput_items_per_second,
        pr.memory_peak_mb,
        pr.created_at
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    """
    
    params = []
    where_clauses = []
    
    if model_name:
        where_clauses.append("m.model_name LIKE ?")
        params.append(f"%{model_name}%")
    
    if model_family:
        where_clauses.append("m.model_family = ?")
        params.append(model_family)
    
    if hardware_type:
        where_clauses.append("hp.hardware_type = ?")
        params.append(hardware_type)
    
    if test_case:
        where_clauses.append("pr.test_case = ?")
        params.append(test_case)
    
    if batch_size is not None:
        where_clauses.append("pr.batch_size = ?")
        params.append(batch_size)
    
    if precision:
        where_clauses.append("pr.precision = ?")
        params.append(precision)
    
    if since:
        try:
            since_date = datetime.datetime.fromisoformat(since.replace('Z', '+00:00'))
            where_clauses.append("pr.created_at >= ?")
            params.append(since_date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format for 'since': {since}")
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY pr.created_at DESC LIMIT ?"
    params.append(limit)
    
    # Execute query
    performance_df = execute_query(query, params)
    
    # Convert to list of dicts
    performance_list = performance_df.to_dict(orient='records')
    return performance_list

@app.get("/api/compatibility", response_model=List[CompatibilityResult])
async def get_compatibility(
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
    hardware_type: Optional[str] = None,
    is_compatible: Optional[bool] = None,
    since: Optional[str] = None,
    limit: int = 100
):
    """Get hardware compatibility results"""
    # Build query with filters
    query = """
    SELECT 
        hc.compatibility_id,
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        hc.is_compatible,
        hc.compatibility_score,
        hc.error_message,
        hc.created_at
    FROM 
        hardware_compatibility hc
    JOIN 
        models m ON hc.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    """
    
    params = []
    where_clauses = []
    
    if model_name:
        where_clauses.append("m.model_name LIKE ?")
        params.append(f"%{model_name}%")
    
    if model_family:
        where_clauses.append("m.model_family = ?")
        params.append(model_family)
    
    if hardware_type:
        where_clauses.append("hp.hardware_type = ?")
        params.append(hardware_type)
    
    if is_compatible is not None:
        where_clauses.append("hc.is_compatible = ?")
        params.append(is_compatible)
    
    if since:
        try:
            since_date = datetime.datetime.fromisoformat(since.replace('Z', '+00:00'))
            where_clauses.append("hc.created_at >= ?")
            params.append(since_date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format for 'since': {since}")
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY hc.created_at DESC LIMIT ?"
    params.append(limit)
    
    # Execute query
    compatibility_df = execute_query(query, params)
    
    # Convert to list of dicts
    compatibility_list = compatibility_df.to_dict(orient='records')
    return compatibility_list

@app.get("/api/tests", response_model=List[TestResult])
async def get_tests(
    model_name: Optional[str] = None,
    hardware_type: Optional[str] = None,
    test_module: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100
):
    """Get integration test results"""
    # Build query with filters
    query = """
    SELECT 
        itr.test_result_id,
        itr.test_module,
        itr.test_class,
        itr.test_name,
        itr.status,
        m.model_name,
        hp.hardware_type,
        itr.error_message,
        itr.created_at
    FROM 
        integration_test_results itr
    LEFT JOIN 
        models m ON itr.model_id = m.model_id
    LEFT JOIN 
        hardware_platforms hp ON itr.hardware_id = hp.hardware_id
    """
    
    params = []
    where_clauses = []
    
    if model_name:
        where_clauses.append("m.model_name LIKE ?")
        params.append(f"%{model_name}%")
    
    if hardware_type:
        where_clauses.append("hp.hardware_type = ?")
        params.append(hardware_type)
    
    if test_module:
        where_clauses.append("itr.test_module LIKE ?")
        params.append(f"%{test_module}%")
    
    if status:
        where_clauses.append("itr.status = ?")
        params.append(status)
    
    if since:
        try:
            since_date = datetime.datetime.fromisoformat(since.replace('Z', '+00:00'))
            where_clauses.append("itr.created_at >= ?")
            params.append(since_date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format for 'since': {since}")
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY itr.created_at DESC LIMIT ?"
    params.append(limit)
    
    # Execute query
    tests_df = execute_query(query, params)
    
    # Convert to list of dicts
    tests_list = tests_df.to_dict(orient='records')
    return tests_list
    
@app.get("/api/web-platform", response_model=List[WebPlatformResult])
async def get_web_platform_results(
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    browser: Optional[str] = None,
    platform: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
):
    """Get web platform test results"""
    # Build query with filters
    query = """
    SELECT 
        wpr.result_id,
        wpr.model_name,
        wpr.model_type,
        wpr.browser,
        wpr.platform,
        wpr.status,
        wpr.execution_time,
        wpr.metrics,
        wpr.error_message,
        wpr.source_file,
        wpr.timestamp
    FROM 
        web_platform_results wpr
    """
    
    params = []
    where_clauses = []
    
    if model_name:
        where_clauses.append("wpr.model_name LIKE ?")
        params.append(f"%{model_name}%")
    
    if model_type:
        where_clauses.append("wpr.model_type = ?")
        params.append(model_type)
    
    if browser:
        where_clauses.append("wpr.browser = ?")
        params.append(browser)
    
    if platform:
        where_clauses.append("wpr.platform = ?")
        params.append(platform)
    
    if status:
        where_clauses.append("wpr.status = ?")
        params.append(status)
    
    if start_date:
        try:
            start_dt = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            where_clauses.append("wpr.timestamp >= ?")
            params.append(start_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format for 'start_date': {start_date}")
    
    if end_date:
        try:
            end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            where_clauses.append("wpr.timestamp <= ?")
            params.append(end_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format for 'end_date': {end_date}")
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY wpr.timestamp DESC LIMIT ?"
    params.append(limit)
    
    # Execute query
    try:
        results_df = execute_query(query, params)
        
        # Parse JSON metrics column
        for i, row in results_df.iterrows():
            if pd.notna(row.get('metrics')):
                try:
                    results_df.at[i, 'metrics'] = json.loads(row['metrics'])
                except:
                    results_df.at[i, 'metrics'] = {}
        
        # Convert to list of dicts
        results_list = results_df.to_dict(orient='records')
        return results_list
    except Exception as e:
        # Check if the error is because the table doesn't exist yet
        if "no such table" in str(e).lower():
            # Return empty list if table doesn't exist
            return []
        else:
            # Re-raise other exceptions
            raise

class WebPlatformResultInput(BaseModel):
    model_name: str
    model_type: str
    browser: str
    platform: str
    status: str
    execution_time: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    source_file: Optional[str] = None
    timestamp: Optional[datetime.datetime] = None

@app.post("/api/web-platform", response_model=Dict[str, Any])
async def add_web_platform_result(result: WebPlatformResultInput):
    """Store a web platform test result in the database"""
    try:
        # Get database connection
        db = get_db()
        
        # Check if web_platform_results table exists, create it if not
        table_exists = db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='web_platform_results'
        """).fetchone()
        
        if not table_exists:
            # Create table
            db.execute("""
            CREATE TABLE web_platform_results (
                result_id INTEGER PRIMARY KEY,
                model_name VARCHAR,
                model_type VARCHAR,
                browser VARCHAR,
                platform VARCHAR,
                status VARCHAR,
                execution_time DOUBLE,
                metrics JSON,
                error_message VARCHAR,
                source_file VARCHAR,
                timestamp TIMESTAMP
            )
            """)
            
            # Create indices
            db.execute("CREATE INDEX idx_wpr_model_type ON web_platform_results(model_type)")
            db.execute("CREATE INDEX idx_wpr_browser ON web_platform_results(browser)")
            db.execute("CREATE INDEX idx_wpr_platform ON web_platform_results(platform)")
            db.execute("CREATE INDEX idx_wpr_timestamp ON web_platform_results(timestamp)")
            
            logger.info("Created web_platform_results table")
        
        # Set default timestamp if not provided
        timestamp = result.timestamp or datetime.datetime.now()
        
        # Convert metrics to JSON string if provided
        metrics_json = None
        if result.metrics is not None:
            metrics_json = json.dumps(result.metrics)
        
        # Insert into database
        inserted_id = db.execute("""
            INSERT INTO web_platform_results (
                model_name, model_type, browser, platform, status, 
                execution_time, metrics, error_message, source_file, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING result_id
        """, (
            result.model_name, result.model_type, result.browser, result.platform, result.status,
            result.execution_time, metrics_json, result.error_message, result.source_file, timestamp
        )).fetchone()[0]
        
        return {"success": True, "result_id": inserted_id}
    except Exception as e:
        logger.error(f"Error storing web platform result: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing result: {str(e)}")

@app.get("/api/charts/compatibility-matrix")
async def get_compatibility_matrix_chart():
    """Generate a compatibility matrix chart"""
    try:
        # Query compatibility data
        query = """
        SELECT 
            m.model_family,
            hp.hardware_type,
            AVG(hc.compatibility_score) as avg_score
        FROM 
            hardware_compatibility hc
        JOIN 
            models m ON hc.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON hc.hardware_id = hp.hardware_id
        GROUP BY 
            m.model_family, hp.hardware_type
        ORDER BY 
            m.model_family, hp.hardware_type
        """
        
        df = execute_query(query)
        
        if df.empty:
            # Return a placeholder image if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No compatibility data available", ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            # Create pivot table
            pivot_df = df.pivot(index='model_family', columns='hardware_type', values='avg_score')
            
            # Fill NaN values with 0
            pivot_df = pivot_df.fillna(0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot_df.values, cmap='RdYlGn', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Compatibility Score', rotation=-90, va="bottom")
            
            # Set axis labels
            ax.set_xticks(np.arange(len(pivot_df.columns)))
            ax.set_yticks(np.arange(len(pivot_df.index)))
            ax.set_xticklabels(pivot_df.columns)
            ax.set_yticklabels(pivot_df.index)
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            for i in range(len(pivot_df.index)):
                for j in range(len(pivot_df.columns)):
                    if not np.isnan(pivot_df.values[i, j]):
                        text = ax.text(j, i, f"{pivot_df.values[i, j]:.2f}",
                                      ha="center", va="center", color="black")
            
            # Add title
            ax.set_title("Model Family - Hardware Compatibility Matrix")
            
            fig.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        plt.close(fig)
        img_buf.seek(0)
        
        return HTMLResponse(content=f'<img src="data:image/png;base64,{base64.b64encode(img_buf.read()).decode()}"/>')
    
    except Exception as e:
        logger.error(f"Error generating compatibility matrix chart: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@app.get("/api/charts/performance-overview")
async def get_performance_overview_chart():
    """Generate a performance overview chart"""
    try:
        # Query performance data
        query = """
        SELECT 
            m.model_family,
            hp.hardware_type,
            AVG(pr.throughput_items_per_second) as avg_throughput
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        GROUP BY 
            m.model_family, hp.hardware_type
        ORDER BY 
            m.model_family, hp.hardware_type
        """
        
        df = execute_query(query)
        
        if df.empty:
            # Return a placeholder image if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No performance data available", ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            # Set up color mapping for hardware types
            hardware_types = df['hardware_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(hardware_types)))
            color_map = dict(zip(hardware_types, colors))
            
            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get unique model families
            model_families = df['model_family'].unique()
            x = np.arange(len(model_families))
            width = 0.8 / len(hardware_types)
            
            # Plot each hardware type as a group of bars
            for i, hw_type in enumerate(hardware_types):
                hw_data = df[df['hardware_type'] == hw_type]
                throughputs = []
                
                for family in model_families:
                    family_data = hw_data[hw_data['model_family'] == family]
                    if not family_data.empty:
                        throughputs.append(family_data['avg_throughput'].values[0])
                    else:
                        throughputs.append(0)
                
                offset = (i - len(hardware_types)/2 + 0.5) * width
                ax.bar(x + offset, throughputs, width, label=hw_type, color=color_map[hw_type])
            
            # Set labels and title
            ax.set_xlabel('Model Family')
            ax.set_ylabel('Average Throughput (items/sec)')
            ax.set_title('Average Throughput by Model Family and Hardware')
            ax.set_xticks(x)
            ax.set_xticklabels(model_families)
            ax.legend()
            
            fig.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        plt.close(fig)
        img_buf.seek(0)
        
        return HTMLResponse(content=f'<img src="data:image/png;base64,{base64.b64encode(img_buf.read()).decode()}"/>')
    
    except Exception as e:
        logger.error(f"Error generating performance overview chart: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@app.get("/api/charts/hardware-comparison")
async def get_hardware_comparison_chart(
    metric: str = "throughput_items_per_second",
    model_family: Optional[str] = None
):
    """Generate a hardware comparison chart"""

@app.get("/api/charts/web-platform-comparison")
async def get_web_platform_comparison_chart(
    model_type: Optional[str] = None,
    browser: Optional[str] = None
):
    """Generate a web platform comparison chart for audio models"""
    try:
        # Build query with filters
        query = """
        SELECT 
            wpr.model_name,
            wpr.model_type,
            wpr.platform,
            wpr.browser,
            AVG(wpr.execution_time) as avg_execution_time
        FROM 
            web_platform_results wpr
        WHERE
            wpr.status = 'successful'
        """
        
        params = []
        conditions = []
        
        if model_type:
            conditions.append("wpr.model_type = ?")
            params.append(model_type)
        
        if browser:
            conditions.append("wpr.browser = ?")
            params.append(browser)
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += """
        GROUP BY 
            wpr.model_name, wpr.model_type, wpr.platform, wpr.browser
        ORDER BY 
            wpr.model_type, wpr.model_name, wpr.platform
        """
        
        try:
            df = execute_query(query, params if params else None)
        except Exception as e:
            # Check if the error is because the table doesn't exist yet
            if "no such table" in str(e).lower():
                # Create an empty dataframe if table doesn't exist
                df = pd.DataFrame(columns=['model_name', 'model_type', 'platform', 'browser', 'avg_execution_time'])
            else:
                # Re-raise other exceptions
                raise
        
        if df.empty:
            # Return a placeholder image if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No web platform test data available", ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            # Create figure with multiple subplots for different model types
            model_types = df['model_type'].unique()
            fig, axes = plt.subplots(len(model_types), 1, figsize=(12, 6 * len(model_types)), squeeze=False)
            
            # Plot each model type in a separate subplot
            for i, mt in enumerate(model_types):
                model_data = df[df['model_type'] == mt]
                
                # Create pivot table for this model type
                platforms = model_data['platform'].unique()
                model_names = model_data['model_name'].unique()
                
                # Prepare data for grouped bar chart
                plot_data = {}
                for platform in platforms:
                    platform_data = model_data[model_data['platform'] == platform]
                    times = []
                    
                    for model in model_names:
                        model_platform_data = platform_data[platform_data['model_name'] == model]
                        if not model_platform_data.empty:
                            times.append(model_platform_data['avg_execution_time'].values[0] * 1000)  # Convert to ms
                        else:
                            times.append(0)
                    
                    plot_data[platform] = times
                
                # Plot grouped bar chart
                ax = axes[i, 0]
                x = np.arange(len(model_names))
                width = 0.8 / len(platforms)
                
                for j, (platform, times) in enumerate(plot_data.items()):
                    offset = (j - len(platforms)/2 + 0.5) * width
                    ax.bar(x + offset, times, width, label=platform.upper())
                
                # Add labels and title
                ax.set_xlabel('Model')
                ax.set_ylabel('Execution Time (ms)')
                ax.set_title(f'Web Platform Performance Comparison: {mt.upper()} Models')
                ax.set_xticks(x)
                ax.set_xticklabels(model_names)
                ax.legend(title='Platform')
                
                # Add value annotations on bars
                for j, (platform, times) in enumerate(plot_data.items()):
                    offset = (j - len(platforms)/2 + 0.5) * width
                    for k, time in enumerate(times):
                        if time > 0:
                            ax.text(x[k] + offset, time + 5, f"{time:.1f}", ha='center', va='bottom', fontsize=8)
            
            fig.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', dpi=150)
        plt.close(fig)
        img_buf.seek(0)
        
        return HTMLResponse(content=f'<img src="data:image/png;base64,{base64.b64encode(img_buf.read()).decode()}"/>')
    
    except Exception as e:
        logger.error(f"Error generating web platform comparison chart: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@app.get("/api/charts/hardware-comparison")
async def get_hardware_comparison_chart(
    metric: str = "throughput_items_per_second",
    model_family: Optional[str] = None
):
    """Generate a hardware comparison chart"""
    try:
        # Map metric parameter to column name
        metric_col = metric
        if metric == "throughput":
            metric_col = "throughput_items_per_second"
        elif metric == "latency":
            metric_col = "average_latency_ms"
        elif metric == "memory":
            metric_col = "memory_peak_mb"
        
        # Build query with filters
        query = f"""
        SELECT 
            m.model_family,
            hp.hardware_type,
            AVG(pr.{metric_col}) as avg_value
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        """
        
        params = []
        where_clauses = []
        
        if model_family:
            where_clauses.append("m.model_family = ?")
            params.append(model_family)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += f"""
        GROUP BY 
            m.model_family, hp.hardware_type
        ORDER BY 
            m.model_family, hp.hardware_type
        """
        
        df = execute_query(query, params if params else None)
        
        if df.empty:
            # Return a placeholder image if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No performance data available", ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            # Create pivot table
            pivot_df = df.pivot(index='model_family', columns='hardware_type', values='avg_value')
            
            # Fill NaN values with 0
            pivot_df = pivot_df.fillna(0)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            pivot_df.plot(kind='bar', ax=ax)
            
            # Set labels and title
            metric_name = metric.replace('_', ' ').title()
            ax.set_xlabel('Model Family')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Average {metric_name} by Model Family and Hardware')
            ax.legend(title='Hardware')
            
            fig.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        plt.close(fig)
        img_buf.seek(0)
        
        return HTMLResponse(content=f'<img src="data:image/png;base64,{base64.b64encode(img_buf.read()).decode()}"/>')
    
    except Exception as e:
        logger.error(f"Error generating hardware comparison chart: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

def main():
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Set global database path
    global db_path
    db_path = args.db
    
    # Check if database file exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)
    
    # Start the server if requested
    if args.serve:
        uvicorn.run("benchmark_db_api:app", host=args.host, port=args.port, reload=args.reload)
    else:
        # Just check if we can connect to the database
        try:
            conn = duckdb.connect(db_path)
            logger.info(f"Successfully connected to database: {db_path}")
            conn.close()
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()