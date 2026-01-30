#!/usr/bin/env python3
"""
FastAPI server for API Management UI

This module provides a FastAPI server that exposes the API Management UI 
functionality via RESTful endpoints for integration with the distributed
testing framework and other components.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import anyio
import uvicorn

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI components
try:
    from api_management_ui import PredictiveAnalyticsUI
    from api_monitoring_dashboard import APIMonitoringDashboard
    from run_api_management_ui import generate_sample_data, save_sample_data
except ImportError:
    # Handle import path issues
    from test.api_management_ui import PredictiveAnalyticsUI
    from test.api_monitoring_dashboard import APIMonitoringDashboard
    from test.run_api_management_ui import generate_sample_data, save_sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="API Management UI Server",
    description="FastAPI server for the API Management UI component of the IPFS Accelerate Distributed Testing Framework",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
router = APIRouter(prefix="/api", tags=["api"])

# Create WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def broadcast(self, client_id: str, message: Dict):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json(message)

manager = ConnectionManager()

# Pydantic models for API requests and responses
class MetricData(BaseModel):
    timestamp: str
    value: float

class DataFilterRequest(BaseModel):
    api_name: str = Field(..., description="Name of the API provider")
    metric_type: str = Field(..., description="Type of metric to retrieve")
    start_date: Optional[str] = Field(None, description="Start date in ISO format")
    end_date: Optional[str] = Field(None, description="End date in ISO format")

class ExportRequest(BaseModel):
    format: str = Field(..., description="Export format (html, png, svg, json, csv, pdf)")
    data_type: str = Field(..., description="Type of data to export (forecast, anomaly, pattern, recommendation)")
    api_name: Optional[str] = Field(None, description="Name of the API provider")
    metric_type: Optional[str] = Field(None, description="Type of metric to export")

class DashboardStatus(BaseModel):
    status: str
    connected_to_monitoring: bool
    data_source: Optional[str] = None
    metrics_count: int
    apis_count: int
    last_update: str

# Global state
dashboard = None
monitoring_dashboard = None
data_path = None
db_repository = None
db_path = None
last_update = datetime.datetime.now().isoformat()

# Initialize the dashboard
def initialize_dashboard(
    monitoring_dashboard_instance: Optional[APIMonitoringDashboard] = None,
    data_file_path: Optional[str] = None,
    generate_sample: bool = False,
    sample_path: str = "./sample_api_data.json",
    database_path: Optional[str] = None,
    generate_db_sample: bool = False,
    theme: str = "cosmo",
    enable_caching: bool = False
) -> PredictiveAnalyticsUI:
    """Initialize the dashboard with optional data sources."""
    global dashboard, monitoring_dashboard, data_path, db_repository, db_path, last_update
    
    # Generate sample data if requested
    if generate_sample:
        save_sample_data(sample_path)
        data_file_path = sample_path
    
    # Set up monitoring dashboard connection if provided
    if monitoring_dashboard_instance:
        monitoring_dashboard = monitoring_dashboard_instance
    
    # Set up database connection if provided
    if database_path:
        db_path = database_path
        try:
            # Import DuckDBAPIMetricsRepository
            try:
                from data.duckdb.api_management import DuckDBAPIMetricsRepository
            except ImportError:
                from test.duckdb_api.api_management import DuckDBAPIMetricsRepository
                
            # Create repository
            db_repository = DuckDBAPIMetricsRepository(
                db_path=database_path,
                create_if_missing=True
            )
            
            # Generate sample data if requested
            if generate_db_sample:
                logger.info(f"Generating sample data in DuckDB database at {database_path}")
                
                # Call the sample data generation method using the module directly
                import subprocess
                import sys
                cmd = [sys.executable, "duckdb_api/api_management/duckdb_api_metrics.py", 
                       "--db-path", database_path, "--generate-sample"]
                
                try:
                    subprocess.run(cmd, check=True)
                    logger.info("Sample data generation completed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error generating sample data: {e}")
            
            logger.info(f"Connected to DuckDB database at {database_path}")
        except Exception as e:
            logger.error(f"Error connecting to DuckDB database: {e}")
            db_repository = None
        
    # Initialize UI
    dashboard = PredictiveAnalyticsUI(
        monitoring_dashboard=monitoring_dashboard,
        data_path=data_file_path,
        theme=theme,
        debug=True,
        enable_caching=enable_caching,
        db_path=db_path,
        db_repository=db_repository
    )
    
    data_path = data_file_path
    last_update = datetime.datetime.now().isoformat()
    
    return dashboard

# API routes
@router.get("/status")
async def get_status() -> DashboardStatus:
    """Get the current status of the dashboard."""
    global dashboard, monitoring_dashboard, data_path, db_repository, db_path, last_update
    
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    
    # Count available metrics and APIs
    metrics_count = 0
    apis_count = 0
    
    if hasattr(dashboard, 'historical_data'):
        metrics_count = len(dashboard.historical_data.keys())
        
        # Count unique APIs across all metrics
        api_set = set()
        for metric in dashboard.historical_data:
            for api in dashboard.historical_data[metric]:
                api_set.add(api)
        apis_count = len(api_set)
    
    # Determine data source
    data_source = None
    if monitoring_dashboard is not None:
        data_source = "monitoring_dashboard"
    elif db_path is not None:
        data_source = f"database:{db_path}"
    elif data_path is not None:
        data_source = f"file:{data_path}"
    
    return DashboardStatus(
        status="running",
        connected_to_monitoring=monitoring_dashboard is not None,
        data_source=data_source,
        metrics_count=metrics_count,
        apis_count=apis_count,
        last_update=last_update
    )

@router.get("/metrics")
async def get_metrics() -> List[str]:
    """Get all available metrics."""
    global dashboard
    
    if not dashboard or not hasattr(dashboard, 'historical_data'):
        raise HTTPException(status_code=503, detail="Dashboard not initialized or no data available")
    
    return list(dashboard.historical_data.keys())

@router.get("/providers")
async def get_providers() -> List[str]:
    """Get all available API providers."""
    global dashboard
    
    if not dashboard or not hasattr(dashboard, 'historical_data'):
        raise HTTPException(status_code=503, detail="Dashboard not initialized or no data available")
    
    # Collect all unique API providers
    api_set = set()
    for metric in dashboard.historical_data:
        for api in dashboard.historical_data[metric]:
            api_set.add(api)
    
    return list(api_set)

@router.get("/data/{metric}/{provider}")
async def get_data(
    metric: str, 
    provider: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> List[Dict]:
    """Get data for a specific metric and provider."""
    global dashboard
    
    if not dashboard or not hasattr(dashboard, 'historical_data'):
        raise HTTPException(status_code=503, detail="Dashboard not initialized or no data available")
    
    if metric not in dashboard.historical_data or provider not in dashboard.historical_data[metric]:
        raise HTTPException(status_code=404, detail=f"No data available for {provider} - {metric}")
    
    data = dashboard.historical_data[metric][provider]
    
    # Filter by date range if provided
    if start_date or end_date:
        filtered_data = []
        for item in data:
            ts = item.get("timestamp")
            if ts:
                if start_date and ts < start_date:
                    continue
                if end_date and ts > end_date:
                    continue
            filtered_data.append(item)
        return filtered_data
    
    return data

@router.get("/anomalies/{metric}/{provider}")
async def get_anomalies(
    metric: str, 
    provider: str, 
    sensitivity: float = Query(0.5, ge=0.0, le=1.0),
    types: Optional[List[str]] = Query(None, description="Anomaly types to include")
) -> List[Dict]:
    """Get anomalies for a specific metric and provider."""
    global dashboard
    
    if not dashboard or not hasattr(dashboard, 'anomalies'):
        raise HTTPException(status_code=503, detail="Dashboard not initialized or no anomalies available")
    
    if metric not in dashboard.anomalies or provider not in dashboard.anomalies[metric]:
        raise HTTPException(status_code=404, detail=f"No anomalies available for {provider} - {metric}")
    
    anomalies = dashboard.anomalies[metric][provider]
    
    # Apply sensitivity threshold
    threshold = 1.0 - sensitivity
    filtered_anomalies = [a for a in anomalies if a.get('confidence', 0) >= threshold]
    
    # Filter by anomaly type if requested
    if types:
        filtered_anomalies = [a for a in filtered_anomalies if a.get('type') in types]
    
    return filtered_anomalies

@router.get("/recommendations/{provider}")
async def get_recommendations(provider: str) -> List[Dict]:
    """Get recommendations for a specific provider."""
    global dashboard
    
    if not dashboard or not hasattr(dashboard, 'recommendations'):
        raise HTTPException(status_code=503, detail="Dashboard not initialized or no recommendations available")
    
    if provider not in dashboard.recommendations:
        raise HTTPException(status_code=404, detail=f"No recommendations available for {provider}")
    
    return dashboard.recommendations[provider]

@router.post("/filter")
async def filter_data(request: DataFilterRequest) -> Dict[str, Any]:
    """Filter data based on provided criteria."""
    global dashboard
    
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    
    result = {
        "historical_data": None,
        "predictions": None,
        "anomalies": None
    }
    
    # Get historical data
    if (hasattr(dashboard, 'historical_data') and 
        request.metric_type in dashboard.historical_data and 
        request.api_name in dashboard.historical_data[request.metric_type]):
        result["historical_data"] = dashboard.historical_data[request.metric_type][request.api_name]
    
    # Get predictions
    if (hasattr(dashboard, 'predictions') and 
        request.metric_type in dashboard.predictions and 
        request.api_name in dashboard.predictions[request.metric_type]):
        result["predictions"] = dashboard.predictions[request.metric_type][request.api_name]
    
    # Get anomalies
    if (hasattr(dashboard, 'anomalies') and 
        request.metric_type in dashboard.anomalies and 
        request.api_name in dashboard.anomalies[request.metric_type]):
        result["anomalies"] = dashboard.anomalies[request.metric_type][request.api_name]
    
    # Filter by date range if provided
    if request.start_date or request.end_date:
        for data_type in result:
            if result[data_type]:
                filtered_data = []
                for item in result[data_type]:
                    ts = item.get("timestamp")
                    if ts:
                        if request.start_date and ts < request.start_date:
                            continue
                        if request.end_date and ts > request.end_date:
                            continue
                    filtered_data.append(item)
                result[data_type] = filtered_data
    
    return result

@router.post("/export")
async def export_data(request: ExportRequest):
    """Export data in the specified format."""
    global dashboard
    
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    
    # Generate export data
    export_data = {
        "format": request.format,
        "data_type": request.data_type,
        "api_name": request.api_name,
        "metric_type": request.metric_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "data": {}
    }
    
    # Collect data based on type
    if request.data_type == "forecast":
        if request.api_name and request.metric_type:
            if (hasattr(dashboard, 'historical_data') and 
                request.metric_type in dashboard.historical_data and 
                request.api_name in dashboard.historical_data[request.metric_type]):
                export_data["data"]["historical"] = dashboard.historical_data[request.metric_type][request.api_name]
            
            if (hasattr(dashboard, 'predictions') and 
                request.metric_type in dashboard.predictions and 
                request.api_name in dashboard.predictions[request.metric_type]):
                export_data["data"]["predictions"] = dashboard.predictions[request.metric_type][request.api_name]
    
    elif request.data_type == "anomaly":
        if request.api_name and request.metric_type:
            if (hasattr(dashboard, 'anomalies') and 
                request.metric_type in dashboard.anomalies and 
                request.api_name in dashboard.anomalies[request.metric_type]):
                export_data["data"]["anomalies"] = dashboard.anomalies[request.metric_type][request.api_name]
    
    elif request.data_type == "recommendation":
        if request.api_name:
            if hasattr(dashboard, 'recommendations') and request.api_name in dashboard.recommendations:
                export_data["data"]["recommendations"] = dashboard.recommendations[request.api_name]
    
    # Return the appropriate response based on format
    if request.format == "json":
        return JSONResponse(content=export_data)
    
    # For other formats, we'd typically generate the file and return it
    # This is a simplified implementation
    return JSONResponse(content={
        "status": "success",
        "message": f"Data exported in {request.format} format",
        "data": export_data
    })

@router.get("/refresh")
async def refresh_data():
    """Refresh data from monitoring dashboard if connected."""
    global dashboard, monitoring_dashboard, last_update
    
    if not dashboard:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    
    if not monitoring_dashboard:
        raise HTTPException(status_code=400, detail="Not connected to a monitoring dashboard")
    
    try:
        dashboard._load_data_from_dashboard()
        last_update = datetime.datetime.now().isoformat()
        
        return {"status": "success", "message": "Data refreshed successfully"}
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Wait for any message from client
            await websocket.receive_text()
            
            # Send current status
            if dashboard:
                metrics_count = 0
                apis_count = 0
                
                if hasattr(dashboard, 'historical_data'):
                    metrics_count = len(dashboard.historical_data.keys())
                    
                    # Count unique APIs across all metrics
                    api_set = set()
                    for metric in dashboard.historical_data:
                        for api in dashboard.historical_data[metric]:
                            api_set.add(api)
                    apis_count = len(api_set)
                
                await websocket.send_json({
                    "status": "running",
                    "connected_to_monitoring": monitoring_dashboard is not None,
                    "metrics_count": metrics_count,
                    "apis_count": apis_count,
                    "last_update": last_update
                })
            else:
                await websocket.send_json({
                    "status": "not_initialized"
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

# Register the router
app.include_router(router)

# Main entry point
def main():
    """Main function to parse arguments and start the server."""
    parser = argparse.ArgumentParser(description='API Management UI FastAPI Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host address to bind to')
    parser.add_argument('--data', type=str, help='Path to JSON data file')
    parser.add_argument('--generate-sample', action='store_true', help='Generate sample data')
    parser.add_argument('--sample-path', type=str, default='./sample_api_data.json', 
                       help='Path to save/load sample data')
    parser.add_argument('--connect-dashboard', action='store_true', help='Connect to live monitoring dashboard')
    parser.add_argument('--db-path', type=str, help='Path to DuckDB database')
    parser.add_argument('--db-generate-sample', action='store_true',
                       help='Generate sample data in the DuckDB database')
    parser.add_argument('--theme', type=str, default='cosmo', 
                       help='UI theme (cosmo, darkly, flatly, etc.)')
    parser.add_argument('--enable-caching', action='store_true',
                       help='Enable data caching for improved performance')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Handle sample data generation
    if args.generate_sample:
        save_sample_data(args.sample_path)
        logger.info(f"Sample data generated and saved to {args.sample_path}")
        if not args.data:
            args.data = args.sample_path
    
    # Connect to dashboard if requested
    dashboard_instance = None
    if args.connect_dashboard:
        try:
            logger.info("Connecting to API monitoring dashboard...")
            dashboard_instance = APIMonitoringDashboard(enable_predictive_analytics=True)
            logger.info("Connected to dashboard successfully")
        except Exception as e:
            logger.error(f"Error connecting to dashboard: {e}")
            logger.info("Falling back to file-based data")
    
    # Initialize the dashboard
    initialize_dashboard(
        monitoring_dashboard_instance=dashboard_instance,
        data_file_path=args.data,
        generate_sample=args.generate_sample,
        sample_path=args.sample_path,
        database_path=args.db_path,
        generate_db_sample=args.db_generate_sample,
        theme=args.theme,
        enable_caching=args.enable_caching
    )
    
    # Start the server
    logger.info(f"Starting API Management UI FastAPI server on {args.host}:{args.port}...")
    if args.debug:
        log_level = "debug"
    else:
        log_level = "info"
    
    uvicorn.run(
        "api_management_ui_server:app", 
        host=args.host, 
        port=args.port, 
        log_level=log_level,
        reload=args.debug
    )

if __name__ == "__main__":
    main()