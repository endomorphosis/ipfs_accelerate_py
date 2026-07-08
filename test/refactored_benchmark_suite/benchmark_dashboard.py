#!/usr/bin/env python3
"""
Benchmark Interactive Dashboard

This module provides a web-based dashboard for visualizing benchmark results.
It integrates with the benchmark API server to display results and provide
interactive filtering, comparison, and analysis tools.
"""

import os
import sys
import json
import time
import argparse
import logging
import datetime
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization libraries
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback, dash_table
    import dash_bootstrap_components as dbc
    import plotly.graph_objs as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: Dashboard dependencies not installed. Run: "
          "pip install dash dash_bootstrap_components plotly pandas")
    print("For real-time updates, also install: pip install Flask-SocketIO requests websocket-client")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("benchmark_dashboard.log")]
)
logger = logging.getLogger("benchmark_dashboard")

# Default settings
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_DB_PATH = "./benchmark_db.duckdb"
DEFAULT_PORT = 8050
DEFAULT_REFRESH_INTERVAL = 10  # seconds

class BenchmarkDataManager:
    """
    Manager for fetching and processing benchmark data.
    
    This class handles data retrieval from the API server and database,
    and provides methods for querying and filtering the data.
    """
    
    def __init__(self, api_url: str = DEFAULT_API_URL, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the data manager.
        
        Args:
            api_url: URL of the benchmark API server
            db_path: Path to the DuckDB database file
        """
        self.api_url = api_url.rstrip('/')
        self.db_path = db_path
        self.active_runs = {}
        self.cached_models = None
        self.cached_hardware = None
        self.cached_reports = None
        self.benchmark_results = None
        self.last_refresh = 0
        
        # Try to connect to DuckDB if available
        self.db_available = False
        try:
            import duckdb
            self.conn = duckdb.connect(db_path)
            self.db_available = True
            logger.info(f"Connected to DuckDB database at {db_path}")
        except ImportError:
            logger.warning("DuckDB is not installed. Database queries will be disabled.")
        except Exception as e:
            logger.warning(f"Failed to connect to DuckDB database: {e}")
        
        # Initialize requests session
        import requests
        self.session = requests.Session()
        
    def refresh_data(self, force: bool = False) -> bool:
        """
        Refresh data from the API server.
        
        Args:
            force: Force refresh even if recently refreshed
            
        Returns:
            True if refresh successful, False otherwise
        """
        now = time.time()
        if not force and now - self.last_refresh < DEFAULT_REFRESH_INTERVAL:
            return True
        
        try:
            # Refresh models
            response = self.session.get(f"{self.api_url}/api/benchmark/models")
            self.cached_models = response.json()
            
            # Refresh hardware
            response = self.session.get(f"{self.api_url}/api/benchmark/hardware")
            self.cached_hardware = response.json()
            
            # Refresh reports
            response = self.session.get(f"{self.api_url}/api/benchmark/reports")
            self.cached_reports = response.json()
            
            # Refresh active runs
            for run_id in list(self.active_runs.keys()):
                try:
                    response = self.session.get(f"{self.api_url}/api/benchmark/status/{run_id}")
                    self.active_runs[run_id] = response.json()
                except Exception as e:
                    logger.warning(f"Failed to get status for run {run_id}: {e}")
            
            # Update last refresh time
            self.last_refresh = now
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh data: {e}")
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get available models.
        
        Returns:
            List of model information
        """
        if self.cached_models is None:
            self.refresh_data(force=True)
        return self.cached_models or []
    
    def get_hardware(self) -> List[Dict[str, Any]]:
        """
        Get available hardware platforms.
        
        Returns:
            List of hardware information
        """
        if self.cached_hardware is None:
            self.refresh_data(force=True)
        return self.cached_hardware or []
    
    def get_reports(self) -> List[Dict[str, Any]]:
        """
        Get available benchmark reports.
        
        Returns:
            List of report information
        """
        if self.cached_reports is None:
            self.refresh_data(force=True)
        return self.cached_reports or []
    
    def get_active_runs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get active benchmark runs.
        
        Returns:
            Dictionary mapping run IDs to run data
        """
        return self.active_runs
    
    def add_run(self, run_id: str, run_data: Dict[str, Any]) -> None:
        """
        Add a run to the active runs list.
        
        Args:
            run_id: ID of the run
            run_data: Run data
        """
        self.active_runs[run_id] = run_data
    
    def remove_run(self, run_id: str) -> None:
        """
        Remove a run from the active runs list.
        
        Args:
            run_id: ID of the run
        """
        if run_id in self.active_runs:
            del self.active_runs[run_id]
    
    def query_results(self, 
                     model: Optional[str] = None, 
                     hardware: Optional[str] = None,
                     batch_size: Optional[int] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query benchmark results from the API.
        
        Args:
            model: Filter by model name
            hardware: Filter by hardware type
            batch_size: Filter by batch size
            limit: Maximum number of results to return
            
        Returns:
            List of benchmark results
        """
        try:
            params = {}
            if model:
                params["model"] = model
            if hardware:
                params["hardware"] = hardware
            if batch_size:
                params["batch_size"] = batch_size
            
            params["limit"] = limit
            
            response = self.session.get(f"{self.api_url}/api/benchmark/query", params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to query results: {e}")
            return []
    
    def query_database(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a custom query on the DuckDB database.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results as a list of dictionaries
        """
        if not self.db_available:
            logger.warning("DuckDB is not available. Cannot execute query.")
            return []
        
        try:
            result = self.conn.execute(query).fetchdf()
            return result.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to execute database query: {e}")
            return []
    
    def get_results_dataframe(self, 
                            model: Optional[str] = None, 
                            hardware: Optional[str] = None,
                            batch_size: Optional[int] = None,
                            limit: int = 1000) -> pd.DataFrame:
        """
        Get benchmark results as a pandas DataFrame.
        
        Args:
            model: Filter by model name
            hardware: Filter by hardware type
            batch_size: Filter by batch size
            limit: Maximum number of results to return
            
        Returns:
            DataFrame containing benchmark results
        """
        results = self.query_results(model, hardware, batch_size, limit)
        
        if not results:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'model_name', 'model_family', 'hardware_type', 'device_name', 
                'test_case', 'batch_size', 'precision', 'throughput_items_per_second',
                'average_latency_ms', 'memory_peak_mb', 'created_at'
            ])
        
        return pd.DataFrame(results)
    
    def get_model_families(self) -> List[str]:
        """
        Get unique model families from the database.
        
        Returns:
            List of model family names
        """
        if not self.db_available:
            # Try to get from the API instead
            models = self.get_models()
            return sorted(list(set(model.get('family', '') for model in models if model.get('family'))))
        
        try:
            result = self.conn.execute("SELECT DISTINCT model_family FROM models WHERE model_family IS NOT NULL").fetchdf()
            return sorted(result['model_family'].tolist())
        except Exception as e:
            logger.error(f"Failed to get model families: {e}")
            return []
    
    def get_hardware_types(self) -> List[str]:
        """
        Get unique hardware types from the database.
        
        Returns:
            List of hardware type names
        """
        if not self.db_available:
            # Try to get from the API instead
            hardware = self.get_hardware()
            return sorted(list(set(hw.get('name', '') for hw in hardware if hw.get('name'))))
        
        try:
            result = self.conn.execute("SELECT DISTINCT hardware_type FROM hardware_platforms").fetchdf()
            return sorted(result['hardware_type'].tolist())
        except Exception as e:
            logger.error(f"Failed to get hardware types: {e}")
            return []
    
    def get_batch_sizes(self) -> List[int]:
        """
        Get unique batch sizes from the database.
        
        Returns:
            List of batch sizes
        """
        if not self.db_available:
            # Return default batch sizes
            return [1, 2, 4, 8, 16, 32, 64]
        
        try:
            result = self.conn.execute("SELECT DISTINCT batch_size FROM performance_results WHERE batch_size IS NOT NULL").fetchdf()
            return sorted(result['batch_size'].tolist())
        except Exception as e:
            logger.error(f"Failed to get batch sizes: {e}")
            return [1, 2, 4, 8, 16, 32, 64]
    
    def get_performance_comparison(self, 
                                 model_families: List[str], 
                                 hardware_types: List[str],
                                 metric: str = "throughput_items_per_second") -> pd.DataFrame:
        """
        Get performance comparison data for visualization.
        
        Args:
            model_families: List of model families to include
            hardware_types: List of hardware types to include
            metric: Performance metric to compare
            
        Returns:
            DataFrame containing performance comparison data
        """
        if not self.db_available:
            # Try to use the API
            results = []
            
            for model_family in model_families:
                for hardware_type in hardware_types:
                    models = [m.get('name') for m in self.get_models() if m.get('family') == model_family]
                    
                    for model in models:
                        model_results = self.query_results(model=model, hardware=hardware_type)
                        
                        for result in model_results:
                            result['model_family'] = model_family
                            results.append(result)
            
            if not results:
                return pd.DataFrame()
                
            return pd.DataFrame(results)
        
        # If we have database access, use a more efficient query
        try:
            # Construct the model family and hardware type filters
            model_family_filter = ", ".join([f"'{family}'" for family in model_families])
            hardware_type_filter = ", ".join([f"'{hw}'" for hw in hardware_types])
            
            # Construct the query
            query = f"""
            SELECT 
                m.model_family,
                h.hardware_type,
                AVG(pr.{metric}) as avg_{metric}
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE 
                m.model_family IN ({model_family_filter})
                AND h.hardware_type IN ({hardware_type_filter})
                AND pr.{metric} IS NOT NULL
            GROUP BY 
                m.model_family, h.hardware_type
            ORDER BY 
                m.model_family, h.hardware_type
            """
            
            result = self.conn.execute(query).fetchdf()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get performance comparison: {e}")
            return pd.DataFrame()
    
    def get_batch_size_scaling(self, 
                             model_family: str, 
                             hardware_type: str,
                             metric: str = "throughput_items_per_second") -> pd.DataFrame:
        """
        Get batch size scaling data for visualization.
        
        Args:
            model_family: Model family to analyze
            hardware_type: Hardware type to analyze
            metric: Performance metric to compare
            
        Returns:
            DataFrame containing batch size scaling data
        """
        if not self.db_available:
            # Try to use the API
            results = []
            batch_sizes = self.get_batch_sizes()
            
            for batch_size in batch_sizes:
                models = [m.get('name') for m in self.get_models() if m.get('family') == model_family]
                
                for model in models:
                    model_results = self.query_results(model=model, hardware=hardware_type, batch_size=batch_size)
                    results.extend(model_results)
            
            if not results:
                return pd.DataFrame()
                
            df = pd.DataFrame(results)
            return df.groupby('batch_size')[metric].mean().reset_index()
        
        # If we have database access, use a more efficient query
        try:
            # Construct the query
            query = f"""
            SELECT 
                pr.batch_size,
                AVG(pr.{metric}) as avg_{metric}
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE 
                m.model_family = '{model_family}'
                AND h.hardware_type = '{hardware_type}'
                AND pr.{metric} IS NOT NULL
                AND pr.batch_size IS NOT NULL
            GROUP BY 
                pr.batch_size
            ORDER BY 
                pr.batch_size
            """
            
            result = self.conn.execute(query).fetchdf()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get batch size scaling: {e}")
            return pd.DataFrame()
    
    def get_top_models(self, 
                     hardware_type: str, 
                     metric: str = "throughput_items_per_second", 
                     limit: int = 10) -> pd.DataFrame:
        """
        Get top-performing models for a specific hardware type.
        
        Args:
            hardware_type: Hardware type to analyze
            metric: Performance metric to compare
            limit: Maximum number of models to return
            
        Returns:
            DataFrame containing top-performing models
        """
        if not self.db_available:
            # Try to use the API
            models = self.get_models()
            results = []
            
            for model in models:
                model_results = self.query_results(model=model.get('name'), hardware=hardware_type)
                results.extend(model_results)
            
            if not results:
                return pd.DataFrame()
                
            df = pd.DataFrame(results)
            
            # Group by model and calculate average metric
            grouped = df.groupby('model_name')[metric].mean().reset_index()
            
            # Sort by metric and take top N
            return grouped.sort_values(metric, ascending=False).head(limit)
        
        # If we have database access, use a more efficient query
        try:
            # Construct the query
            query = f"""
            SELECT 
                m.model_name,
                AVG(pr.{metric}) as avg_{metric}
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE 
                h.hardware_type = '{hardware_type}'
                AND pr.{metric} IS NOT NULL
            GROUP BY 
                m.model_name
            ORDER BY 
                avg_{metric} DESC
            LIMIT {limit}
            """
            
            result = self.conn.execute(query).fetchdf()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get top models: {e}")
            return pd.DataFrame()
    
    def get_hardware_comparison(self, 
                              model_name: str, 
                              metric: str = "throughput_items_per_second") -> pd.DataFrame:
        """
        Get hardware comparison data for a specific model.
        
        Args:
            model_name: Model to analyze
            metric: Performance metric to compare
            
        Returns:
            DataFrame containing hardware comparison data
        """
        if not self.db_available:
            # Try to use the API
            hardware_types = self.get_hardware_types()
            results = []
            
            for hardware_type in hardware_types:
                hardware_results = self.query_results(model=model_name, hardware=hardware_type)
                results.extend(hardware_results)
            
            if not results:
                return pd.DataFrame()
                
            df = pd.DataFrame(results)
            
            # Group by hardware type and calculate average metric
            return df.groupby('hardware_type')[metric].mean().reset_index()
        
        # If we have database access, use a more efficient query
        try:
            # Construct the query
            query = f"""
            SELECT 
                h.hardware_type,
                AVG(pr.{metric}) as avg_{metric}
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE 
                m.model_name = '{model_name}'
                AND pr.{metric} IS NOT NULL
            GROUP BY 
                h.hardware_type
            ORDER BY 
                avg_{metric} DESC
            """
            
            result = self.conn.execute(query).fetchdf()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get hardware comparison: {e}")
            return pd.DataFrame()


class BenchmarkDashboard:
    """
    Interactive dashboard for benchmark visualization.
    
    This class creates and manages a Dash application for visualizing
    benchmark results with interactive filters and charts.
    """
    
    def __init__(self, 
                api_url: str = DEFAULT_API_URL, 
                db_path: str = DEFAULT_DB_PATH,
                port: int = DEFAULT_PORT,
                debug: bool = False):
        """
        Initialize the dashboard.
        
        Args:
            api_url: URL of the benchmark API server
            db_path: Path to the DuckDB database file
            port: Port to run the dashboard server on
            debug: Enable debug mode
        """
        self.api_url = api_url
        self.db_path = db_path
        self.port = port
        self.debug = debug
        
        # Create data manager
        self.data_manager = BenchmarkDataManager(api_url, db_path)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Benchmark Dashboard",
            suppress_callback_exceptions=True
        )
        
        # Define layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
    
    def _create_layout(self) -> html.Div:
        """
        Create the dashboard layout.
        
        Returns:
            Dash layout
        """
        # Sidebar with filters
        sidebar = dbc.Card([
            dbc.CardHeader("Filters"),
            dbc.CardBody([
                html.H5("Model Family"),
                dcc.Dropdown(
                    id="model-family-filter",
                    options=[],
                    multi=True,
                    placeholder="Select model families..."
                ),
                html.Hr(),
                
                html.H5("Hardware Type"),
                dcc.Dropdown(
                    id="hardware-type-filter",
                    options=[],
                    multi=True,
                    placeholder="Select hardware types..."
                ),
                html.Hr(),
                
                html.H5("Batch Size"),
                dcc.Dropdown(
                    id="batch-size-filter",
                    options=[],
                    multi=True,
                    placeholder="Select batch sizes..."
                ),
                html.Hr(),
                
                html.H5("Metric"),
                dcc.Dropdown(
                    id="metric-selector",
                    options=[
                        {"label": "Throughput (items/s)", "value": "throughput_items_per_second"},
                        {"label": "Latency (ms)", "value": "average_latency_ms"},
                        {"label": "Memory Usage (MB)", "value": "memory_peak_mb"}
                    ],
                    value="throughput_items_per_second",
                    clearable=False
                ),
                html.Hr(),
                
                dbc.Button("Apply Filters", id="apply-filters-button", color="primary", className="w-100 mb-2"),
                dbc.Button("Refresh Data", id="refresh-data-button", color="secondary", className="w-100")
            ])
        ], className="mb-4 h-100")
        
        # Main content area with tabs
        main_content = html.Div([
            dbc.Tabs([
                dbc.Tab([
                    html.Div([
                        html.H3("Overview", className="text-center mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Hardware Comparison"),
                                    dbc.CardBody([
                                        dcc.Dropdown(
                                            id="overview-model-selector",
                                            placeholder="Select a model...",
                                            options=[]
                                        ),
                                        dcc.Graph(id="hardware-comparison-chart")
                                    ])
                                ])
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Top Models"),
                                    dbc.CardBody([
                                        dcc.Dropdown(
                                            id="top-models-hardware-selector",
                                            placeholder="Select hardware type...",
                                            options=[]
                                        ),
                                        dcc.Graph(id="top-models-chart")
                                    ])
                                ])
                            ], width=6)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Batch Size Scaling"),
                                    dbc.CardBody([
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Dropdown(
                                                    id="scaling-model-family-selector",
                                                    placeholder="Select model family...",
                                                    options=[]
                                                )
                                            ], width=6),
                                            dbc.Col([
                                                dcc.Dropdown(
                                                    id="scaling-hardware-selector",
                                                    placeholder="Select hardware type...",
                                                    options=[]
                                                )
                                            ], width=6)
                                        ], className="mb-2"),
                                        dcc.Graph(id="batch-scaling-chart")
                                    ])
                                ])
                            ], width=12)
                        ])
                    ], className="p-3")
                ], label="Overview"),
                
                dbc.Tab([
                    html.Div([
                        html.H3("Performance Comparison", className="text-center mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Hardware/Model Family Performance Heatmap"),
                            dbc.CardBody([
                                dcc.Graph(id="performance-heatmap")
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Detailed Results"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id="results-table",
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textOverflow": "ellipsis",
                                        "overflow": "hidden",
                                        "maxWidth": 0
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold"
                                    },
                                    filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    column_selectable="single",
                                    page_action="native",
                                    export_format="csv"
                                )
                            ])
                        ])
                    ], className="p-3")
                ], label="Comparison"),
                
                dbc.Tab([
                    html.Div([
                        html.H3("Active Benchmarks", className="text-center mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Live Benchmark Runs"),
                            dbc.CardBody([
                                html.Div(id="active-runs-container")
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Start New Benchmark"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Priority"),
                                        dcc.Dropdown(
                                            id="new-benchmark-priority",
                                            options=[
                                                {"label": "Critical", "value": "critical"},
                                                {"label": "High", "value": "high"},
                                                {"label": "Medium", "value": "medium"},
                                                {"label": "All", "value": "all"}
                                            ],
                                            value="high"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Hardware"),
                                        dcc.Dropdown(
                                            id="new-benchmark-hardware",
                                            options=[],
                                            multi=True,
                                            value=["cpu"]
                                        )
                                    ], width=6)
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Models (optional)"),
                                        dcc.Dropdown(
                                            id="new-benchmark-models",
                                            options=[],
                                            multi=True
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Batch Sizes"),
                                        dcc.Dropdown(
                                            id="new-benchmark-batch-sizes",
                                            options=[
                                                {"label": str(size), "value": size}
                                                for size in [1, 2, 4, 8, 16, 32, 64]
                                            ],
                                            multi=True,
                                            value=[1, 8]
                                        )
                                    ], width=6)
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Precision"),
                                        dcc.Dropdown(
                                            id="new-benchmark-precision",
                                            options=[
                                                {"label": "FP32", "value": "fp32"},
                                                {"label": "FP16", "value": "fp16"},
                                                {"label": "INT8", "value": "int8"}
                                            ],
                                            value="fp32"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Progressive Mode", "value": "progressive_mode"},
                                                {"label": "Incremental", "value": "incremental"}
                                            ],
                                            value=["progressive_mode", "incremental"],
                                            id="new-benchmark-options",
                                            switch=True,
                                            className="mt-4"
                                        )
                                    ], width=6)
                                ], className="mb-3"),
                                
                                dbc.Button("Start Benchmark", id="start-benchmark-button", color="success", className="w-100")
                            ])
                        ])
                    ], className="p-3")
                ], label="Live Runs"),
                
                dbc.Tab([
                    html.Div([
                        html.H3("Reports", className="text-center mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Available Reports"),
                            dbc.CardBody([
                                html.Div(id="reports-container")
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Custom SQL Query"),
                            dbc.CardBody([
                                dbc.Textarea(
                                    id="sql-query-input",
                                    placeholder="Enter SQL query...",
                                    rows=6,
                                    className="mb-3"
                                ),
                                dbc.Button("Run Query", id="run-query-button", color="primary", className="w-100 mb-3"),
                                html.Div(id="query-results-container")
                            ])
                        ])
                    ], className="p-3")
                ], label="Reports"),
                
                dbc.Tab([
                    html.Div([
                        html.H3("Help & Documentation", className="text-center mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Dashboard Usage"),
                            dbc.CardBody([
                                html.H4("Overview"),
                                html.P("""
                                    This dashboard provides visualization and analysis tools for benchmark results. 
                                    Use the sidebar filters to select specific model families, hardware types, and metrics to analyze.
                                """),
                                
                                html.H4("Tabs"),
                                html.Ul([
                                    html.Li(html.B("Overview"), " - High-level overview of performance across different dimensions"),
                                    html.Li(html.B("Comparison"), " - Detailed comparison of performance metrics with heatmap visualization"),
                                    html.Li(html.B("Live Runs"), " - Monitor active benchmark runs and start new benchmarks"),
                                    html.Li(html.B("Reports"), " - View benchmark reports and run custom SQL queries")
                                ]),
                                
                                html.H4("Metrics"),
                                html.Ul([
                                    html.Li(html.B("Throughput"), " - Items processed per second (higher is better)"),
                                    html.Li(html.B("Latency"), " - Average processing time in milliseconds (lower is better)"),
                                    html.Li(html.B("Memory Usage"), " - Peak memory usage in megabytes (lower is better)")
                                ])
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("API Reference"),
                            dbc.CardBody([
                                html.H4("API Endpoints"),
                                html.Ul([
                                    html.Li(html.Code("POST /api/benchmark/run"), " - Start a benchmark run"),
                                    html.Li(html.Code("GET /api/benchmark/status/{run_id}"), " - Get status of a benchmark run"),
                                    html.Li(html.Code("GET /api/benchmark/results/{run_id}"), " - Get results of a completed benchmark"),
                                    html.Li(html.Code("GET /api/benchmark/models"), " - List available models"),
                                    html.Li(html.Code("GET /api/benchmark/hardware"), " - List available hardware platforms"),
                                    html.Li(html.Code("GET /api/benchmark/reports"), " - List available benchmark reports"),
                                    html.Li(html.Code("GET /api/benchmark/query"), " - Query benchmark results"),
                                    html.Li(html.Code("WebSocket /api/benchmark/ws/{run_id}"), " - Real-time benchmark updates")
                                ]),
                                
                                html.H4("Database Schema"),
                                html.P("The benchmark results are stored in a DuckDB database with the following tables:"),
                                html.Ul([
                                    html.Li(html.Code("models"), " - Model information"),
                                    html.Li(html.Code("hardware_platforms"), " - Hardware platform information"),
                                    html.Li(html.Code("test_runs"), " - Test run information"),
                                    html.Li(html.Code("performance_results"), " - Performance results"),
                                    html.Li(html.Code("hardware_compatibility"), " - Hardware compatibility information")
                                ])
                            ])
                        ])
                    ], className="p-3")
                ], label="Help")
            ], id="main-tabs")
        ])
        
        # Footer
        footer = html.Footer([
            html.Hr(),
            html.P(
                "Benchmark Dashboard - Connected to API: " + self.api_url,
                className="text-center text-muted"
            )
        ])
        
        # Combine everything into the final layout
        layout = dbc.Container([
            html.H1("Benchmark Results Dashboard", className="my-4 text-center"),
            
            # Main layout with sidebar and content
            dbc.Row([
                dbc.Col(sidebar, width=3),
                dbc.Col(main_content, width=9)
            ]),
            
            footer,
            
            # Hidden divs for storing data
            html.Div(id="refresh-trigger", style={"display": "none"}),
            dcc.Interval(id="auto-refresh-interval", interval=DEFAULT_REFRESH_INTERVAL * 1000, n_intervals=0),
            
            # WebSocket connection management (dummy component for callback)
            html.Div(id="websocket-management", style={"display": "none"})
            
        ], fluid=True)
        
        return layout
    
    def _register_callbacks(self) -> None:
        """Register all dashboard callbacks."""
        
        # Initialize filters on page load
        @self.app.callback(
            [
                Output("model-family-filter", "options"),
                Output("hardware-type-filter", "options"),
                Output("batch-size-filter", "options"),
                Output("overview-model-selector", "options"),
                Output("top-models-hardware-selector", "options"),
                Output("scaling-model-family-selector", "options"),
                Output("scaling-hardware-selector", "options"),
                Output("new-benchmark-hardware", "options"),
                Output("new-benchmark-models", "options")
            ],
            [Input("refresh-trigger", "children")]
        )
        def initialize_filters(trigger):
            # Get model families
            model_families = self.data_manager.get_model_families()
            model_family_options = [{"label": family, "value": family} for family in model_families]
            
            # Get hardware types
            hardware_types = self.data_manager.get_hardware_types()
            hardware_options = [{"label": hw, "value": hw} for hw in hardware_types]
            
            # Get batch sizes
            batch_sizes = self.data_manager.get_batch_sizes()
            batch_size_options = [{"label": str(size), "value": size} for size in batch_sizes]
            
            # Get models for overview
            models = self.data_manager.get_models()
            model_options = [{"label": model.get("name", ""), "value": model.get("name", "")} for model in models]
            
            return (
                model_family_options,
                hardware_options,
                batch_size_options,
                model_options,
                hardware_options,
                model_family_options,
                hardware_options,
                hardware_options,
                model_options
            )
        
        # Refresh data
        @self.app.callback(
            Output("refresh-trigger", "children"),
            [Input("refresh-data-button", "n_clicks"), 
             Input("auto-refresh-interval", "n_intervals")]
        )
        def refresh_data(n_clicks, n_intervals):
            self.data_manager.refresh_data(force=True if n_clicks else False)
            return datetime.datetime.now().isoformat()
        
        # Update performance heatmap
        @self.app.callback(
            Output("performance-heatmap", "figure"),
            [Input("apply-filters-button", "n_clicks")],
            [
                State("model-family-filter", "value"),
                State("hardware-type-filter", "value"),
                State("metric-selector", "value")
            ]
        )
        def update_performance_heatmap(n_clicks, model_families, hardware_types, metric):
            # If filters are empty, use defaults
            if not model_families or len(model_families) == 0:
                model_families = self.data_manager.get_model_families()[:5]  # Take first 5 families
            
            if not hardware_types or len(hardware_types) == 0:
                hardware_types = self.data_manager.get_hardware_types()
            
            # Get data
            df = self.data_manager.get_performance_comparison(model_families, hardware_types, metric)
            
            if df.empty:
                return {
                    "data": [],
                    "layout": {
                        "title": "No data available",
                        "annotations": [{
                            "text": "No data available for the selected filters",
                            "showarrow": False,
                            "font": {"size": 20}
                        }]
                    }
                }
            
            # Create pivot table for heatmap
            metric_col = f"avg_{metric}"
            if metric_col not in df.columns:
                if "avg_throughput_items_per_second" in df.columns:
                    metric_col = "avg_throughput_items_per_second"
                elif len(df.columns) >= 3:
                    # Assume the third column is the metric
                    metric_col = df.columns[2]
                else:
                    return {
                        "data": [],
                        "layout": {
                            "title": "No data available",
                            "annotations": [{
                                "text": "No data available for the selected filters",
                                "showarrow": False,
                                "font": {"size": 20}
                            }]
                        }
                    }
            
            # Create pivot table for heatmap
            if "model_family" in df.columns and "hardware_type" in df.columns:
                pivot_df = df.pivot(index="model_family", columns="hardware_type", values=metric_col)
            else:
                # Use available columns for pivot
                columns = df.columns.tolist()
                if len(columns) >= 3:
                    pivot_df = df.pivot(index=columns[0], columns=columns[1], values=columns[2])
                else:
                    return {
                        "data": [],
                        "layout": {
                            "title": "No data available",
                            "annotations": [{
                                "text": "No data available for the selected filters",
                                "showarrow": False,
                                "font": {"size": 20}
                            }]
                        }
                    }
            
            # Create heatmap
            metric_label = "Throughput (items/s)" if "throughput" in metric else "Latency (ms)" if "latency" in metric else "Memory (MB)"
            title = f"Performance Heatmap: {metric_label}"
            
            # Create a custom colorscale based on the metric
            if "latency" in metric or "memory" in metric:
                # For latency and memory, lower is better (red to green)
                colorscale = [
                    [0, "green"],
                    [0.5, "yellow"],
                    [1, "red"]
                ]
                # Invert the values for proper coloring
                z_data = -pivot_df.values
            else:
                # For throughput, higher is better (green to red)
                colorscale = [
                    [0, "red"],
                    [0.5, "yellow"],
                    [1, "green"]
                ]
                z_data = pivot_df.values
            
            return {
                "data": [{
                    "type": "heatmap",
                    "z": z_data,
                    "x": pivot_df.columns.tolist(),
                    "y": pivot_df.index.tolist(),
                    "colorscale": colorscale,
                    "colorbar": {"title": metric_label}
                }],
                "layout": {
                    "title": title,
                    "xaxis": {"title": "Hardware Type"},
                    "yaxis": {"title": "Model Family"}
                }
            }
        
        # Update results table
        @self.app.callback(
            Output("results-table", "data"),
            Output("results-table", "columns"),
            [Input("apply-filters-button", "n_clicks")],
            [
                State("model-family-filter", "value"),
                State("hardware-type-filter", "value"),
                State("batch-size-filter", "value")
            ]
        )
        def update_results_table(n_clicks, model_families, hardware_types, batch_sizes):
            # Query results based on filters
            results = []
            
            # If filters are empty, use some reasonable defaults to avoid returning too much data
            if (not model_families or len(model_families) == 0) and (not hardware_types or len(hardware_types) == 0):
                # Limit to a few recent results
                df = self.data_manager.get_results_dataframe(limit=100)
            else:
                # Apply filters one by one to get combined results
                for model_family in (model_families or []):
                    models = [m.get("name") for m in self.data_manager.get_models() if m.get("family") == model_family]
                    
                    for model in models:
                        for hardware_type in (hardware_types or self.data_manager.get_hardware_types()):
                            # Query with batch size filter if provided
                            if batch_sizes and len(batch_sizes) > 0:
                                for batch_size in batch_sizes:
                                    model_results = self.data_manager.query_results(
                                        model=model, 
                                        hardware=hardware_type,
                                        batch_size=batch_size
                                    )
                                    results.extend(model_results)
                            else:
                                # Query without batch size filter
                                model_results = self.data_manager.query_results(
                                    model=model, 
                                    hardware=hardware_type
                                )
                                results.extend(model_results)
                
                df = pd.DataFrame(results)
            
            # If dataframe is empty, return empty data
            if df.empty:
                return [], []
            
            # Format columns for DataTable
            columns = [
                {"name": col, "id": col} for col in df.columns
            ]
            
            # Return data and columns
            return df.to_dict("records"), columns
        
        # Update hardware comparison chart
        @self.app.callback(
            Output("hardware-comparison-chart", "figure"),
            [Input("overview-model-selector", "value"),
             Input("metric-selector", "value")]
        )
        def update_hardware_comparison(model, metric):
            if not model:
                return {
                    "data": [],
                    "layout": {
                        "title": "Select a model to see hardware comparison",
                        "xaxis": {"title": "Hardware Type"},
                        "yaxis": {"title": "Performance"}
                    }
                }
            
            # Get data
            df = self.data_manager.get_hardware_comparison(model, metric)
            
            if df.empty:
                return {
                    "data": [],
                    "layout": {
                        "title": f"No data available for {model}",
                        "annotations": [{
                            "text": "No benchmark data available",
                            "showarrow": False,
                            "font": {"size": 20}
                        }]
                    }
                }
            
            # Create bar chart
            metric_label = "Throughput (items/s)" if "throughput" in metric else "Latency (ms)" if "latency" in metric else "Memory (MB)"
            metric_col = f"avg_{metric}"
            if metric_col not in df.columns and len(df.columns) >= 2:
                metric_col = df.columns[1]  # Assume second column is the metric
            
            if "hardware_type" in df.columns:
                x_col = "hardware_type"
            elif len(df.columns) >= 1:
                x_col = df.columns[0]  # Assume first column is the hardware type
            else:
                return {
                    "data": [],
                    "layout": {
                        "title": f"No data available for {model}",
                        "annotations": [{
                            "text": "No benchmark data available",
                            "showarrow": False,
                            "font": {"size": 20}
                        }]
                    }
                }
            
            return {
                "data": [{
                    "type": "bar",
                    "x": df[x_col].tolist(),
                    "y": df[metric_col].tolist(),
                    "marker": {"color": "rgb(55, 83, 109)"}
                }],
                "layout": {
                    "title": f"Hardware Performance Comparison: {model}",
                    "xaxis": {"title": "Hardware Type"},
                    "yaxis": {"title": metric_label}
                }
            }
        
        # Update top models chart
        @self.app.callback(
            Output("top-models-chart", "figure"),
            [Input("top-models-hardware-selector", "value"),
             Input("metric-selector", "value")]
        )
        def update_top_models(hardware_type, metric):
            if not hardware_type:
                return {
                    "data": [],
                    "layout": {
                        "title": "Select a hardware type to see top models",
                        "xaxis": {"title": "Model"},
                        "yaxis": {"title": "Performance"}
                    }
                }
            
            # Get data
            df = self.data_manager.get_top_models(hardware_type, metric, limit=10)
            
            if df.empty:
                return {
                    "data": [],
                    "layout": {
                        "title": f"No data available for {hardware_type}",
                        "annotations": [{
                            "text": "No benchmark data available",
                            "showarrow": False,
                            "font": {"size": 20}
                        }]
                    }
                }
            
            # Create bar chart
            metric_label = "Throughput (items/s)" if "throughput" in metric else "Latency (ms)" if "latency" in metric else "Memory (MB)"
            metric_col = f"avg_{metric}"
            if metric_col not in df.columns and len(df.columns) >= 2:
                metric_col = df.columns[1]  # Assume second column is the metric
            
            if "model_name" in df.columns:
                x_col = "model_name"
            elif len(df.columns) >= 1:
                x_col = df.columns[0]  # Assume first column is the model name
            else:
                return {
                    "data": [],
                    "layout": {
                        "title": f"No data available for {hardware_type}",
                        "annotations": [{
                            "text": "No benchmark data available",
                            "showarrow": False,
                            "font": {"size": 20}
                        }]
                    }
                }
            
            return {
                "data": [{
                    "type": "bar",
                    "x": df[x_col].tolist(),
                    "y": df[metric_col].tolist(),
                    "marker": {"color": "rgb(26, 118, 255)"}
                }],
                "layout": {
                    "title": f"Top Models for {hardware_type}",
                    "xaxis": {"title": "Model", "tickangle": 45},
                    "yaxis": {"title": metric_label}
                }
            }
        
        # Update batch scaling chart
        @self.app.callback(
            Output("batch-scaling-chart", "figure"),
            [
                Input("scaling-model-family-selector", "value"),
                Input("scaling-hardware-selector", "value"),
                Input("metric-selector", "value")
            ]
        )
        def update_batch_scaling(model_family, hardware_type, metric):
            if not model_family or not hardware_type:
                return {
                    "data": [],
                    "layout": {
                        "title": "Select model family and hardware type",
                        "xaxis": {"title": "Batch Size"},
                        "yaxis": {"title": "Performance"}
                    }
                }
            
            # Get data
            df = self.data_manager.get_batch_size_scaling(model_family, hardware_type, metric)
            
            if df.empty:
                return {
                    "data": [],
                    "layout": {
                        "title": f"No data available for {model_family} on {hardware_type}",
                        "annotations": [{
                            "text": "No benchmark data available",
                            "showarrow": False,
                            "font": {"size": 20}
                        }]
                    }
                }
            
            # Create line chart
            metric_label = "Throughput (items/s)" if "throughput" in metric else "Latency (ms)" if "latency" in metric else "Memory (MB)"
            metric_col = f"avg_{metric}"
            if metric_col not in df.columns and len(df.columns) >= 2:
                metric_col = df.columns[1]  # Assume second column is the metric
            
            if "batch_size" in df.columns:
                x_col = "batch_size"
            elif len(df.columns) >= 1:
                x_col = df.columns[0]  # Assume first column is the batch size
            else:
                return {
                    "data": [],
                    "layout": {
                        "title": f"No data available for {model_family} on {hardware_type}",
                        "annotations": [{
                            "text": "No benchmark data available",
                            "showarrow": False,
                            "font": {"size": 20}
                        }]
                    }
                }
            
            return {
                "data": [{
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": df[x_col].tolist(),
                    "y": df[metric_col].tolist(),
                    "marker": {"size": 10}
                }],
                "layout": {
                    "title": f"Batch Size Scaling: {model_family} on {hardware_type}",
                    "xaxis": {"title": "Batch Size", "type": "log"},
                    "yaxis": {"title": metric_label}
                }
            }
        
        # Update active runs container
        @self.app.callback(
            Output("active-runs-container", "children"),
            [Input("refresh-trigger", "children")]
        )
        def update_active_runs(trigger):
            active_runs = self.data_manager.get_active_runs()
            
            if not active_runs:
                return html.Div([
                    html.P("No active benchmark runs.", className="text-center text-muted my-4")
                ])
            
            run_cards = []
            for run_id, run_data in active_runs.items():
                # Calculate progress bar style
                progress = run_data.get("progress", 0) * 100
                progress_color = "success" if progress == 100 else "info"
                
                # Create progress bar and status info
                run_cards.append(
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5(f"Run ID: {run_id}", className="d-inline"),
                            html.Span(f"Status: {run_data.get('status', 'unknown').upper()}", 
                                     className="float-right badge bg-primary")
                        ]),
                        dbc.CardBody([
                            html.P(f"Current Step: {run_data.get('current_step', 'Unknown')}"),
                            html.P(f"Models: {run_data.get('completed_models', 0)}/{run_data.get('total_models', 0)}"),
                            dbc.Progress(
                                value=progress,
                                color=progress_color,
                                striped=progress < 100,
                                animated=progress < 100,
                                className="mb-3"
                            ),
                            html.P(f"Started: {run_data.get('start_time', 'Unknown')}"),
                            html.P(f"Elapsed: {run_data.get('elapsed_time', 0):.1f} seconds")
                        ]),
                        dbc.CardFooter([
                            dbc.Button("View Details", color="primary", size="sm", 
                                      id={"type": "view-run-button", "index": run_id},
                                      className="mr-2"),
                            dbc.Button("Remove", color="danger", size="sm", 
                                      id={"type": "remove-run-button", "index": run_id})
                        ])
                    ], className="mb-3")
                )
            
            return html.Div(run_cards)
        
        # Update reports container
        @self.app.callback(
            Output("reports-container", "children"),
            [Input("refresh-trigger", "children")]
        )
        def update_reports(trigger):
            reports = self.data_manager.get_reports()
            
            if not reports:
                return html.Div([
                    html.P("No benchmark reports available.", className="text-center text-muted my-4")
                ])
            
            report_cards = []
            for report in reports:
                report_cards.append(
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5(f"Run ID: {report.get('run_id', 'Unknown')}", className="d-inline"),
                            html.Span(f"Status: {report.get('status', 'unknown').upper()}", 
                                     className="float-right badge bg-primary")
                        ]),
                        dbc.CardBody([
                            html.H5("Available Reports:"),
                            html.Ul([
                                html.Li([
                                    html.A(report_file, 
                                           href=f"file://{os.path.join(report.get('path', ''), report_file)}",
                                           target="_blank")
                                ]) for report_file in report.get("reports", [])
                            ])
                        ])
                    ], className="mb-3")
                )
            
            return html.Div(report_cards)
        
        # Start new benchmark
        @self.app.callback(
            Output("start-benchmark-button", "disabled"),
            Output("start-benchmark-button", "children"),
            Input("start-benchmark-button", "n_clicks"),
            [
                State("new-benchmark-priority", "value"),
                State("new-benchmark-hardware", "value"),
                State("new-benchmark-models", "value"),
                State("new-benchmark-batch-sizes", "value"),
                State("new-benchmark-precision", "value"),
                State("new-benchmark-options", "value")
            ]
        )
        def start_new_benchmark(n_clicks, priority, hardware, models, batch_sizes, precision, options):
            if not n_clicks:
                return False, "Start Benchmark"
            
            if not hardware:
                return False, "Start Benchmark"
            
            # Prepare request data
            data = {
                "priority": priority,
                "hardware": hardware,
                "batch_sizes": batch_sizes,
                "precision": precision,
                "progressive_mode": "progressive_mode" in options,
                "incremental": "incremental" in options
            }
            
            if models:
                data["models"] = models
            
            try:
                # Send API request
                import requests
                response = requests.post(f"{self.api_url}/api/benchmark/run", json=data)
                
                if response.status_code == 200:
                    run_data = response.json()
                    run_id = run_data.get("run_id")
                    
                    # Add to active runs
                    self.data_manager.add_run(run_id, run_data)
                    
                    # Connect WebSocket for this run
                    try:
                        import threading
                        import websocket
                        
                        def on_message(ws, message):
                            try:
                                data = json.loads(message)
                                self.data_manager.add_run(run_id, data)
                            except:
                                pass
                        
                        def on_error(ws, error):
                            logger.error(f"WebSocket error for run {run_id}: {error}")
                        
                        def on_close(ws, close_status_code, close_msg):
                            logger.info(f"WebSocket connection closed for run {run_id}")
                        
                        def on_open(ws):
                            logger.info(f"WebSocket connection opened for run {run_id}")
                        
                        def ws_thread():
                            ws_url = f"ws://{self.api_url.split('://', 1)[1]}/api/benchmark/ws/{run_id}"
                            ws = websocket.WebSocketApp(
                                ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close
                            )
                            ws.run_forever()
                        
                        # Start WebSocket connection in a separate thread
                        threading.Thread(target=ws_thread).start()
                        
                    except ImportError:
                        logger.warning("websocket-client package not installed. Will use polling for updates.")
                    
                    # Force a data refresh
                    self.data_manager.refresh_data(force=True)
                    
                    return False, "Benchmark Started"
                else:
                    logger.error(f"Failed to start benchmark: {response.text}")
                    return False, "Failed to Start"
                    
            except Exception as e:
                logger.error(f"Error starting benchmark: {e}")
                return False, "Error"
        
        # Run SQL query
        @self.app.callback(
            Output("query-results-container", "children"),
            Input("run-query-button", "n_clicks"),
            State("sql-query-input", "value")
        )
        def run_sql_query(n_clicks, query):
            if not n_clicks or not query:
                return html.Div()
            
            try:
                # Execute query
                results = self.data_manager.query_database(query)
                
                if not results:
                    return html.Div([
                        html.P("No results returned.", className="text-center text-muted my-4")
                    ])
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Create DataTable
                return html.Div([
                    html.H5(f"Query Results ({len(results)} rows)"),
                    dash_table.DataTable(
                        data=df.to_dict("records"),
                        columns=[{"name": col, "id": col} for col in df.columns],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textOverflow": "ellipsis",
                            "overflow": "hidden",
                            "maxWidth": 0
                        },
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                            "fontWeight": "bold"
                        },
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable="single",
                        page_action="native",
                        export_format="csv"
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                return html.Div([
                    html.Div(
                        f"Error executing query: {str(e)}",
                        className="alert alert-danger"
                    )
                ])
    
    def run(self) -> None:
        """Run the dashboard server."""
        self.app.run_server(
            host="0.0.0.0",
            port=self.port,
            debug=self.debug
        )


def main():
    """Main entry point when run directly."""
    parser = argparse.ArgumentParser(description="Benchmark Dashboard")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL of the benchmark API server")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the DuckDB database file")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run the dashboard server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run the dashboard
    logger.info(f"Starting Benchmark Dashboard on port {args.port}")
    logger.info(f"Connecting to API server at {args.api_url}")
    logger.info(f"Using database at {args.db_path}")
    
    dashboard = BenchmarkDashboard(
        api_url=args.api_url,
        db_path=args.db_path,
        port=args.port,
        debug=args.debug
    )
    
    dashboard.run()


if __name__ == "__main__":
    main()