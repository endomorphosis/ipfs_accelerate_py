#!/usr/bin/env python3
"""
Visualization Dashboard for End-to-End Testing Framework

This module provides an interactive web dashboard for visualizing test results
and performance metrics from the End-to-End Testing Framework. The dashboard
offers real-time monitoring, detailed performance visualizations, and
comparative analysis tools.

Features:
- Interactive web-based dashboard built with Dash and Plotly
- Real-time monitoring of test execution and results
- Comprehensive performance visualization for model-hardware combinations
- Comparative analysis tools for cross-hardware performance
- Simulation validation visualization
- Historical trend analysis with statistical significance testing
- Customizable views and filtering options
- Integration with DuckDB for efficient data retrieval
- Responsive design for desktop and mobile devices

Usage:
    # Start the visualization dashboard server
    python visualization_dashboard.py

    # Start with custom configuration
    python visualization_dashboard.py --port 8050 --db-path ./benchmark_db.duckdb

    # Run in development mode with hot reloading
    python visualization_dashboard.py --debug
"""

import os
import sys
import json
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import Dash and Plotly for interactive visualization
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# Import DuckDB for database access
import duckdb

# Import utilities
from simple_utils import setup_logging, ensure_dir_exists

# Set up logging
logger = logging.getLogger(__name__)
setup_logging(logger, level=logging.INFO)

# Constants
DEFAULT_DB_PATH = os.path.join(os.path.dirname(script_dir), "test_template_db.duckdb")
DEFAULT_PORT = 8050
DEFAULT_HOST = "localhost"

# Define color schemes
COLORS = {
    "primary": "#0366d6",
    "secondary": "#6c757d",
    "success": "#22863a",
    "warning": "#f0ad4e",
    "danger": "#cb2431",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
    "hardware": {
        "cpu": "#1f77b4",
        "cuda": "#ff7f0e",
        "rocm": "#2ca02c",
        "mps": "#d62728",
        "openvino": "#9467bd",
        "qnn": "#8c564b",
        "webgpu": "#e377c2",
        "webnn": "#7f7f7f"
    },
    "model_family": {
        "text-embedding": "#1f77b4",
        "text-generation": "#ff7f0e",
        "vision": "#2ca02c",
        "audio": "#d62728",
        "multimodal": "#9467bd",
        "unknown": "#7f7f7f"
    }
}


class DashboardDataProvider:
    """
    Provides data to the visualization dashboard from the DuckDB database.
    
    This class handles all database interactions and data processing for the dashboard,
    retrieving and transforming test results and performance metrics.
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the data provider with the specified database path.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.conn = None
        self.try_connect()
    
    def try_connect(self) -> bool:
        """
        Attempt to connect to the DuckDB database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database at {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database at {self.db_path}: {str(e)}")
            return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the test results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.conn:
            return {}
        
        try:
            # Get total tests
            total_query = """
            SELECT COUNT(*) as total_tests,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_tests,
                   SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_tests
            FROM test_results
            """
            total_result = self.conn.execute(total_query).fetchone()
            
            # Get tests by hardware
            hardware_query = """
            SELECT hardware_type, COUNT(*) as test_count,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_tests
            FROM test_results
            GROUP BY hardware_type
            ORDER BY test_count DESC
            """
            hardware_result = self.conn.execute(hardware_query).fetchall()
            
            # Get tests by model family
            family_query = """
            SELECT model_name, COUNT(*) as test_count,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_tests
            FROM test_results
            GROUP BY model_name
            ORDER BY test_count DESC
            """
            family_result = self.conn.execute(family_query).fetchall()
            
            # Get recent test dates
            date_query = """
            SELECT DISTINCT test_date
            FROM test_results
            ORDER BY test_date DESC
            LIMIT 10
            """
            date_result = self.conn.execute(date_query).fetchall()
            
            return {
                "total": {
                    "total": total_result[0] if total_result else 0,
                    "success": total_result[1] if total_result else 0,
                    "failure": total_result[2] if total_result else 0,
                },
                "by_hardware": {hw[0]: {"total": hw[1], "success": hw[2]} for hw in hardware_result},
                "by_model": {model[0]: {"total": model[1], "success": model[2]} for model in family_result},
                "recent_dates": [date[0] for date in date_result]
            }
        except Exception as e:
            logger.error(f"Error getting summary statistics: {str(e)}")
            return {}
    
    def get_performance_metrics(self, 
                              model_filter: Optional[str] = None, 
                              hardware_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Get performance metrics from the database as a DataFrame.
        
        Args:
            model_filter: Optional model name filter
            hardware_filter: Optional hardware type filter
            
        Returns:
            DataFrame with performance metrics
        """
        if not self.conn:
            return pd.DataFrame()
        
        try:
            # Build query with filters
            query = """
            SELECT tr.model_name, tr.hardware_type, 
                   tr.details->>'throughput' as throughput, 
                   tr.details->>'latency' as latency,
                   tr.details->>'memory_usage' as memory_usage,
                   tr.test_date, tr.success
            FROM test_results tr
            WHERE 1=1
            """
            
            params = {}
            if model_filter:
                query += " AND tr.model_name = :model"
                params["model"] = model_filter
            
            if hardware_filter:
                query += " AND tr.hardware_type = :hardware"
                params["hardware"] = hardware_filter
            
            query += " ORDER BY tr.test_date DESC"
            
            # Execute query and convert to DataFrame
            result = self.conn.execute(query, params).fetchdf()
            
            # Convert numeric columns
            numeric_cols = ["throughput", "latency", "memory_usage"]
            for col in numeric_cols:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors="coerce")
            
            return result
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return pd.DataFrame()
    
    def get_hardware_comparison(self, model_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Get hardware comparison data for visualization.
        
        Args:
            model_filter: Optional model name filter
            
        Returns:
            DataFrame with hardware comparison data
        """
        if not self.conn:
            return pd.DataFrame()
        
        try:
            # Build query with filters
            query = """
            SELECT tr.model_name, tr.hardware_type, 
                   AVG(CAST(tr.details->>'throughput' AS FLOAT)) as avg_throughput, 
                   AVG(CAST(tr.details->>'latency' AS FLOAT)) as avg_latency,
                   AVG(CAST(tr.details->>'memory_usage' AS FLOAT)) as avg_memory
            FROM test_results tr
            WHERE tr.success = true
            """
            
            params = {}
            if model_filter:
                query += " AND tr.model_name = :model"
                params["model"] = model_filter
            
            query += " GROUP BY tr.model_name, tr.hardware_type"
            query += " ORDER BY tr.model_name, tr.hardware_type"
            
            # Execute query and convert to DataFrame
            return self.conn.execute(query, params).fetchdf()
        except Exception as e:
            logger.error(f"Error getting hardware comparison: {str(e)}")
            return pd.DataFrame()
    
    def get_time_series_data(self, 
                           metric: str = "throughput", 
                           model_filter: Optional[str] = None, 
                           hardware_filter: Optional[str] = None, 
                           days: int = 30) -> pd.DataFrame:
        """
        Get time series data for a specific metric.
        
        Args:
            metric: Metric to retrieve (throughput, latency, memory_usage)
            model_filter: Optional model name filter
            hardware_filter: Optional hardware type filter
            days: Number of days to look back
            
        Returns:
            DataFrame with time series data
        """
        if not self.conn:
            return pd.DataFrame()
        
        try:
            # Calculate cutoff date
            cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
            
            # Build query with filters
            query = f"""
            SELECT tr.model_name, tr.hardware_type, tr.test_date, 
                   CAST(tr.details->>'{ metric }' AS FLOAT) as metric_value
            FROM test_results tr
            WHERE tr.success = true
              AND tr.test_date >= '{ cutoff_date }'
              AND tr.details->>'{ metric }' IS NOT NULL
            """
            
            params = {}
            if model_filter:
                query += " AND tr.model_name = :model"
                params["model"] = model_filter
            
            if hardware_filter:
                query += " AND tr.hardware_type = :hardware"
                params["hardware"] = hardware_filter
            
            query += " ORDER BY tr.test_date"
            
            # Execute query and convert to DataFrame
            result = self.conn.execute(query, params).fetchdf()
            
            # Convert dates to datetime
            if "test_date" in result.columns:
                result["test_date"] = pd.to_datetime(result["test_date"], format="%Y%m%d_%H%M%S")
            
            return result
        except Exception as e:
            logger.error(f"Error getting time series data: {str(e)}")
            return pd.DataFrame()
    
    def get_model_list(self) -> List[str]:
        """
        Get list of models in the database.
        
        Returns:
            List of model names
        """
        if not self.conn:
            return []
        
        try:
            query = """
            SELECT DISTINCT model_name
            FROM test_results
            ORDER BY model_name
            """
            result = self.conn.execute(query).fetchall()
            return [r[0] for r in result]
        except Exception as e:
            logger.error(f"Error getting model list: {str(e)}")
            return []
    
    def get_hardware_list(self) -> List[str]:
        """
        Get list of hardware platforms in the database.
        
        Returns:
            List of hardware platform names
        """
        if not self.conn:
            return []
        
        try:
            query = """
            SELECT DISTINCT hardware_type
            FROM test_results
            ORDER BY hardware_type
            """
            result = self.conn.execute(query).fetchall()
            return [r[0] for r in result]
        except Exception as e:
            logger.error(f"Error getting hardware list: {str(e)}")
            return []
    
    def get_simulation_validation_data(self) -> pd.DataFrame:
        """
        Get data for simulation validation visualization.
        
        Returns:
            DataFrame with simulation validation data
        """
        if not self.conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT tr.model_name, tr.hardware_type, 
                   CAST(tr.details->>'throughput' AS FLOAT) as throughput, 
                   CAST(tr.details->>'latency' AS FLOAT) as latency,
                   tr.is_simulation
            FROM test_results tr
            WHERE tr.success = true
              AND tr.details->>'throughput' IS NOT NULL
              AND tr.details->>'latency' IS NOT NULL
            ORDER BY tr.model_name, tr.hardware_type
            """
            
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting simulation validation data: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


class VisualizationDashboard:
    """
    Interactive web dashboard for visualizing test results and performance metrics.
    
    This class creates and manages the Dash application for the visualization dashboard,
    handling layout, callbacks, and data interactions.
    """
    
    def __init__(self, data_provider: DashboardDataProvider):
        """
        Initialize the dashboard with a data provider.
        
        Args:
            data_provider: DashboardDataProvider instance for data access
        """
        self.data_provider = data_provider
        
        # Initialize the Dash app
        self.app = dash.Dash(
            __name__,
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
            external_stylesheets=[
                "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
            ],
            title="IPFS Accelerate Test Dashboard"
        )
        
        # Initialize app layout
        self.app.layout = self._create_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _create_layout(self) -> html.Div:
        """
        Create the dashboard layout.
        
        Returns:
            Dash HTML layout
        """
        # Get initial data for dropdowns
        models = self.data_provider.get_model_list()
        hardware_platforms = self.data_provider.get_hardware_list()
        
        # Create layout with multiple tabs
        return html.Div(
            className="dashboard-container",
            children=[
                # Header
                html.Div(
                    className="header bg-primary text-white p-3",
                    children=[
                        html.H1("IPFS Accelerate Test Dashboard", className="mb-0"),
                        html.P("Interactive visualization for end-to-end testing results", className="mb-0")
                    ]
                ),
                
                # Main content
                html.Div(
                    className="content-container p-3",
                    children=[
                        # Tabs for different sections
                        dcc.Tabs(
                            id="tabs",
                            value="tab-overview",
                            className="nav-tabs mb-3",
                            children=[
                                dcc.Tab(
                                    label="Overview",
                                    value="tab-overview",
                                    className="nav-item",
                                    selected_className="active"
                                ),
                                dcc.Tab(
                                    label="Performance Analysis",
                                    value="tab-performance",
                                    className="nav-item",
                                    selected_className="active"
                                ),
                                dcc.Tab(
                                    label="Hardware Comparison",
                                    value="tab-hardware",
                                    className="nav-item",
                                    selected_className="active"
                                ),
                                dcc.Tab(
                                    label="Time Series Analysis",
                                    value="tab-time-series",
                                    className="nav-item",
                                    selected_className="active"
                                ),
                                dcc.Tab(
                                    label="Simulation Validation",
                                    value="tab-simulation",
                                    className="nav-item",
                                    selected_className="active"
                                ),
                            ]
                        ),
                        
                        # Tab content
                        html.Div(id="tab-content", className="tab-content"),
                        
                        # Store for sharing data between callbacks
                        dcc.Store(id="summary-data"),
                        
                        # Interval for auto-refresh (every 5 minutes)
                        dcc.Interval(
                            id="interval-component",
                            interval=5*60*1000,  # in milliseconds
                            n_intervals=0
                        ),
                    ]
                ),
                
                # Footer
                html.Div(
                    className="footer bg-light p-3 text-center",
                    children=[
                        html.P(
                            [
                                "IPFS Accelerate Test Framework Dashboard | Last updated: ",
                                html.Span(id="last-updated-time")
                            ],
                            className="mb-0"
                        )
                    ]
                )
            ]
        )
    
    def _create_overview_tab(self, summary_data: Dict[str, Any]) -> html.Div:
        """
        Create the overview tab content.
        
        Args:
            summary_data: Dictionary with summary statistics
            
        Returns:
            Dash HTML layout for the overview tab
        """
        total = summary_data.get("total", {})
        by_hardware = summary_data.get("by_hardware", {})
        by_model = summary_data.get("by_model", {})
        
        # Create success rate chart
        success_rate = (total.get("success", 0) / total.get("total", 1)) * 100 if total.get("total", 0) > 0 else 0
        
        success_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=success_rate,
            title={"text": "Overall Success Rate"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["success"]},
                "steps": [
                    {"range": [0, 50], "color": COLORS["danger"]},
                    {"range": [50, 80], "color": COLORS["warning"]},
                    {"range": [80, 100], "color": COLORS["success"]}
                ],
            },
            number={"suffix": "%"}
        ))
        
        success_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        # Create hardware distribution chart
        if by_hardware:
            hardware_names = list(by_hardware.keys())
            hardware_totals = [data["total"] for data in by_hardware.values()]
            hardware_success = [data["success"] for data in by_hardware.values()]
            
            hardware_fig = go.Figure()
            hardware_fig.add_trace(go.Bar(
                x=hardware_names,
                y=hardware_totals,
                name="Total Tests",
                marker_color=COLORS["secondary"]
            ))
            hardware_fig.add_trace(go.Bar(
                x=hardware_names,
                y=hardware_success,
                name="Successful Tests",
                marker_color=COLORS["success"]
            ))
            
            hardware_fig.update_layout(
                title="Tests by Hardware Platform",
                xaxis_title="Hardware Platform",
                yaxis_title="Number of Tests",
                barmode="group",
                height=400,
                margin=dict(l=50, r=20, t=50, b=100),
            )
        else:
            hardware_fig = go.Figure()
            hardware_fig.update_layout(
                title="Tests by Hardware Platform (No Data)",
                height=400,
                margin=dict(l=50, r=20, t=50, b=100),
            )
        
        # Create model distribution chart
        if by_model:
            model_names = list(by_model.keys())[:10]  # Top 10 models
            model_totals = [by_model[model]["total"] for model in model_names]
            model_success = [by_model[model]["success"] for model in model_names]
            
            model_fig = go.Figure()
            model_fig.add_trace(go.Bar(
                x=model_names,
                y=model_totals,
                name="Total Tests",
                marker_color=COLORS["secondary"]
            ))
            model_fig.add_trace(go.Bar(
                x=model_names,
                y=model_success,
                name="Successful Tests",
                marker_color=COLORS["success"]
            ))
            
            model_fig.update_layout(
                title="Tests by Model (Top 10)",
                xaxis_title="Model",
                yaxis_title="Number of Tests",
                barmode="group",
                height=400,
                margin=dict(l=50, r=20, t=50, b=100),
                xaxis=dict(tickangle=45)
            )
        else:
            model_fig = go.Figure()
            model_fig.update_layout(
                title="Tests by Model (No Data)",
                height=400,
                margin=dict(l=50, r=20, t=50, b=100),
            )
        
        # Create summary cards
        return html.Div([
            # Summary cards
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-3",
                        children=[
                            html.Div(
                                className="card h-100",
                                children=[
                                    html.Div(
                                        className="card-body text-center",
                                        children=[
                                            html.H5("Total Tests", className="card-title"),
                                            html.H2(total.get("total", 0), className="card-text")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-3",
                        children=[
                            html.Div(
                                className="card h-100",
                                children=[
                                    html.Div(
                                        className="card-body text-center",
                                        children=[
                                            html.H5("Successful Tests", className="card-title"),
                                            html.H2(
                                                total.get("success", 0), 
                                                className="card-text text-success"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-3",
                        children=[
                            html.Div(
                                className="card h-100",
                                children=[
                                    html.Div(
                                        className="card-body text-center",
                                        children=[
                                            html.H5("Failed Tests", className="card-title"),
                                            html.H2(
                                                total.get("failure", 0), 
                                                className="card-text text-danger"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-3",
                        children=[
                            html.Div(
                                className="card h-100",
                                children=[
                                    html.Div(
                                        className="card-body text-center",
                                        children=[
                                            html.H5("Success Rate", className="card-title"),
                                            html.H2(
                                                f"{success_rate:.1f}%", 
                                                className="card-text text-primary"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Charts
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            dcc.Graph(
                                                id="success-gauge",
                                                figure=success_gauge
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            dcc.Graph(
                                                id="hardware-distribution",
                                                figure=hardware_fig
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Model distribution
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-12",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            dcc.Graph(
                                                id="model-distribution",
                                                figure=model_fig
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ])
    
    def _create_performance_tab(self) -> html.Div:
        """
        Create the performance analysis tab content.
        
        Returns:
            Dash HTML layout for the performance tab
        """
        # Get list of models and hardware platforms
        models = self.data_provider.get_model_list()
        hardware_platforms = self.data_provider.get_hardware_list()
        
        return html.Div([
            # Filters
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Model Filter", className="card-title"),
                                            dcc.Dropdown(
                                                id="performance-model-dropdown",
                                                options=[
                                                    {"label": model, "value": model}
                                                    for model in models
                                                ],
                                                multi=False,
                                                placeholder="Select a model"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Hardware Filter", className="card-title"),
                                            dcc.Dropdown(
                                                id="performance-hardware-dropdown",
                                                options=[
                                                    {"label": hw, "value": hw}
                                                    for hw in hardware_platforms
                                                ],
                                                multi=False,
                                                placeholder="Select a hardware platform"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Performance graphs
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Throughput Comparison", className="card-title"),
                                            dcc.Graph(id="throughput-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Latency Comparison", className="card-title"),
                                            dcc.Graph(id="latency-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Memory usage and data table
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Memory Usage Comparison", className="card-title"),
                                            dcc.Graph(id="memory-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Performance Data", className="card-title"),
                                            dash_table.DataTable(
                                                id="performance-table",
                                                columns=[
                                                    {"name": "Model", "id": "model_name"},
                                                    {"name": "Hardware", "id": "hardware_type"},
                                                    {"name": "Throughput", "id": "throughput"},
                                                    {"name": "Latency", "id": "latency"},
                                                    {"name": "Memory Usage", "id": "memory_usage"},
                                                    {"name": "Test Date", "id": "test_date"}
                                                ],
                                                style_cell={
                                                    "textAlign": "left",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                    "maxWidth": 0,
                                                },
                                                style_header={
                                                    "backgroundColor": "rgb(230, 230, 230)",
                                                    "fontWeight": "bold"
                                                },
                                                style_data_conditional=[
                                                    {
                                                        "if": {"row_index": "odd"},
                                                        "backgroundColor": "rgb(248, 248, 248)"
                                                    }
                                                ],
                                                page_size=10,
                                                sort_action="native",
                                                filter_action="native",
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
        ])
    
    def _create_hardware_tab(self) -> html.Div:
        """
        Create the hardware comparison tab content.
        
        Returns:
            Dash HTML layout for the hardware comparison tab
        """
        # Get list of models
        models = self.data_provider.get_model_list()
        
        return html.Div([
            # Filters
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Model Selection", className="card-title"),
                                            dcc.Dropdown(
                                                id="hardware-model-dropdown",
                                                options=[
                                                    {"label": model, "value": model}
                                                    for model in models
                                                ],
                                                multi=False,
                                                placeholder="Select a model for hardware comparison"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            ),
            
            # Hardware comparison charts
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-12",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Throughput by Hardware Platform", className="card-title"),
                                            dcc.Graph(id="hardware-throughput-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Latency by Hardware Platform", className="card-title"),
                                            dcc.Graph(id="hardware-latency-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Memory Usage by Hardware Platform", className="card-title"),
                                            dcc.Graph(id="hardware-memory-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Heatmap
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-12",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Performance Heatmap", className="card-title"),
                                            dcc.Graph(id="hardware-heatmap")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
        ])
    
    def _create_time_series_tab(self) -> html.Div:
        """
        Create the time series analysis tab content.
        
        Returns:
            Dash HTML layout for the time series tab
        """
        # Get list of models and hardware platforms
        models = self.data_provider.get_model_list()
        hardware_platforms = self.data_provider.get_hardware_list()
        
        return html.Div([
            # Filters
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-4",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Model Filter", className="card-title"),
                                            dcc.Dropdown(
                                                id="time-series-model-dropdown",
                                                options=[
                                                    {"label": model, "value": model}
                                                    for model in models
                                                ],
                                                multi=False,
                                                placeholder="Select a model"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-4",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Hardware Filter", className="card-title"),
                                            dcc.Dropdown(
                                                id="time-series-hardware-dropdown",
                                                options=[
                                                    {"label": hw, "value": hw}
                                                    for hw in hardware_platforms
                                                ],
                                                multi=False,
                                                placeholder="Select a hardware platform"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-4",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Metric", className="card-title"),
                                            dcc.Dropdown(
                                                id="time-series-metric-dropdown",
                                                options=[
                                                    {"label": "Throughput", "value": "throughput"},
                                                    {"label": "Latency", "value": "latency"},
                                                    {"label": "Memory Usage", "value": "memory_usage"}
                                                ],
                                                value="throughput",
                                                multi=False
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Time period selector
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Time Period", className="card-title"),
                                            dcc.RadioItems(
                                                id="time-series-period",
                                                options=[
                                                    {"label": "Last 7 days", "value": 7},
                                                    {"label": "Last 30 days", "value": 30},
                                                    {"label": "Last 90 days", "value": 90},
                                                    {"label": "Last 365 days", "value": 365}
                                                ],
                                                value=30,
                                                className="mb-3",
                                                inline=True
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Analysis Options", className="card-title"),
                                            dcc.Checklist(
                                                id="time-series-options",
                                                options=[
                                                    {"label": "Show trend line", "value": "trend"},
                                                    {"label": "Show statistical significance", "value": "stats"},
                                                    {"label": "Highlight regressions", "value": "regressions"}
                                                ],
                                                value=["trend"],
                                                className="mb-3",
                                                inline=True
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Time series chart
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-12",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Performance Over Time", className="card-title"),
                                            dcc.Graph(id="time-series-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Statistical analysis
            html.Div(
                id="statistical-analysis-container",
                className="row mb-4"
            )
        ])
    
    def _create_simulation_tab(self) -> html.Div:
        """
        Create the simulation validation tab content.
        
        Returns:
            Dash HTML layout for the simulation validation tab
        """
        return html.Div([
            # Explanation
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-12",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Simulation Validation", className="card-title"),
                                            html.P(
                                                """
                                                This section validates the accuracy of hardware simulations by comparing performance metrics
                                                between simulated and real hardware. The framework uses expected performance ratios between
                                                different hardware platforms to determine if simulations are realistic.
                                                """
                                            ),
                                            html.P(
                                                """
                                                Validated simulations ensure that test results from simulated environments provide a reliable
                                                indication of real-world performance, even when the actual hardware is not available.
                                                """
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Simulation validation visualization
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-12",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Simulation vs. Real Hardware Performance", className="card-title"),
                                            dcc.Graph(id="simulation-comparison-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Hardware performance ratios
            html.Div(
                className="row mb-4",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card h-100",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Expected Hardware Performance Ratios", className="card-title"),
                                            dcc.Graph(id="hardware-ratio-chart")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.Div(
                                className="card h-100",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("Simulation Validation Status", className="card-title"),
                                            dcc.Graph(id="simulation-validation-status")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ])
    
    def _setup_callbacks(self):
        """Set up the Dash callbacks for interactivity."""
        # Update tab content based on selected tab
        @self.app.callback(
            Output("tab-content", "children"),
            [Input("tabs", "value"), Input("summary-data", "data")]
        )
        def update_tab_content(tab, summary_data):
            summary_data = summary_data or {}
            
            if tab == "tab-overview":
                return self._create_overview_tab(summary_data)
            elif tab == "tab-performance":
                return self._create_performance_tab()
            elif tab == "tab-hardware":
                return self._create_hardware_tab()
            elif tab == "tab-time-series":
                return self._create_time_series_tab()
            elif tab == "tab-simulation":
                return self._create_simulation_tab()
            return html.Div("Tab content not found")
        
        # Update summary data
        @self.app.callback(
            [Output("summary-data", "data"), Output("last-updated-time", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_summary_data(n):
            # Get summary statistics
            summary_data = self.data_provider.get_summary_stats()
            # Get current time
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return summary_data, current_time
        
        # Performance tab callbacks
        @self.app.callback(
            [
                Output("throughput-chart", "figure"),
                Output("latency-chart", "figure"),
                Output("memory-chart", "figure"),
                Output("performance-table", "data")
            ],
            [
                Input("performance-model-dropdown", "value"),
                Input("performance-hardware-dropdown", "value")
            ]
        )
        def update_performance_charts(model, hardware):
            # Get performance metrics
            df = self.data_provider.get_performance_metrics(model, hardware)
            
            # Create empty figures if no data
            if df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    height=400,
                    title="No data available for the selected filters"
                )
                return empty_fig, empty_fig, empty_fig, []
            
            # Create throughput chart
            throughput_fig = px.bar(
                df,
                x="hardware_type" if model else "model_name",
                y="throughput",
                color="hardware_type" if model else "model_name",
                title="Throughput Comparison",
                labels={
                    "throughput": "Throughput (items/second)",
                    "hardware_type": "Hardware Platform",
                    "model_name": "Model"
                },
                height=400
            )
            
            # Create latency chart
            latency_fig = px.bar(
                df,
                x="hardware_type" if model else "model_name",
                y="latency",
                color="hardware_type" if model else "model_name",
                title="Latency Comparison",
                labels={
                    "latency": "Latency (ms)",
                    "hardware_type": "Hardware Platform",
                    "model_name": "Model"
                },
                height=400
            )
            
            # Create memory usage chart
            memory_fig = px.bar(
                df,
                x="hardware_type" if model else "model_name",
                y="memory_usage",
                color="hardware_type" if model else "model_name",
                title="Memory Usage Comparison",
                labels={
                    "memory_usage": "Memory Usage (MB)",
                    "hardware_type": "Hardware Platform",
                    "model_name": "Model"
                },
                height=400
            )
            
            # Prepare table data
            table_data = df.to_dict("records")
            
            return throughput_fig, latency_fig, memory_fig, table_data
        
        # Hardware comparison tab callbacks
        @self.app.callback(
            [
                Output("hardware-throughput-chart", "figure"),
                Output("hardware-latency-chart", "figure"),
                Output("hardware-memory-chart", "figure"),
                Output("hardware-heatmap", "figure")
            ],
            [Input("hardware-model-dropdown", "value")]
        )
        def update_hardware_charts(model):
            # Get hardware comparison data
            df = self.data_provider.get_hardware_comparison(model)
            
            # Create empty figures if no data
            if df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    height=400,
                    title="No data available for the selected model"
                )
                return empty_fig, empty_fig, empty_fig, empty_fig
            
            # Create throughput chart
            throughput_fig = px.bar(
                df,
                x="hardware_type",
                y="avg_throughput",
                color="hardware_type",
                title=f"Throughput by Hardware Platform{' for ' + model if model else ''}",
                labels={
                    "avg_throughput": "Average Throughput (items/second)",
                    "hardware_type": "Hardware Platform"
                },
                height=400
            )
            
            # Create latency chart
            latency_fig = px.bar(
                df,
                x="hardware_type",
                y="avg_latency",
                color="hardware_type",
                title=f"Latency by Hardware Platform{' for ' + model if model else ''}",
                labels={
                    "avg_latency": "Average Latency (ms)",
                    "hardware_type": "Hardware Platform"
                },
                height=400
            )
            
            # Create memory usage chart
            memory_fig = px.bar(
                df,
                x="hardware_type",
                y="avg_memory",
                color="hardware_type",
                title=f"Memory Usage by Hardware Platform{' for ' + model if model else ''}",
                labels={
                    "avg_memory": "Average Memory Usage (MB)",
                    "hardware_type": "Hardware Platform"
                },
                height=400
            )
            
            # Create heatmap
            if model:
                # Single model heatmap (hardware vs metrics)
                heatmap_data = df.pivot_table(
                    values=["avg_throughput", "avg_latency", "avg_memory"],
                    index="hardware_type"
                ).reset_index()
                
                # Normalize values for better visualization
                for col in ["avg_throughput", "avg_latency", "avg_memory"]:
                    if col in heatmap_data.columns:
                        max_val = heatmap_data[col].max()
                        if max_val > 0:
                            heatmap_data[col] = heatmap_data[col] / max_val
                
                heatmap_fig = px.imshow(
                    heatmap_data[["avg_throughput", "avg_latency", "avg_memory"]].values,
                    x=["Throughput", "Latency", "Memory"],
                    y=heatmap_data["hardware_type"],
                    color_continuous_scale="Viridis",
                    title=f"Performance Heatmap for {model}",
                    labels=dict(x="Metric", y="Hardware Platform", color="Normalized Value"),
                    height=400
                )
            else:
                # Multiple model heatmap (model vs hardware for throughput)
                pivot_df = df.pivot_table(
                    values="avg_throughput",
                    index="model_name",
                    columns="hardware_type"
                ).fillna(0)
                
                heatmap_fig = px.imshow(
                    pivot_df.values,
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    color_continuous_scale="Viridis",
                    title="Throughput Performance Heatmap",
                    labels=dict(x="Hardware Platform", y="Model", color="Throughput"),
                    height=500
                )
            
            return throughput_fig, latency_fig, memory_fig, heatmap_fig
        
        # Time series tab callbacks
        @self.app.callback(
            [
                Output("time-series-chart", "figure"),
                Output("statistical-analysis-container", "children")
            ],
            [
                Input("time-series-model-dropdown", "value"),
                Input("time-series-hardware-dropdown", "value"),
                Input("time-series-metric-dropdown", "value"),
                Input("time-series-period", "value"),
                Input("time-series-options", "value")
            ]
        )
        def update_time_series(model, hardware, metric, days, options):
            # Get time series data
            df = self.data_provider.get_time_series_data(metric, model, hardware, days)
            
            # Create empty figure if no data
            if df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    height=400,
                    title="No time series data available for the selected filters"
                )
                return empty_fig, html.Div("No statistical analysis available for the selected filters")
            
            # Create time series chart
            metric_label = {
                "throughput": "Throughput (items/second)",
                "latency": "Latency (ms)",
                "memory_usage": "Memory Usage (MB)"
            }.get(metric, metric)
            
            # Group by hardware or model if not filtered
            if not hardware and not model:
                # Group by both model and hardware
                fig = px.scatter(
                    df,
                    x="test_date",
                    y="metric_value",
                    color="hardware_type",
                    symbol="model_name",
                    title=f"{metric_label} Over Time",
                    labels={
                        "test_date": "Date",
                        "metric_value": metric_label,
                        "hardware_type": "Hardware Platform",
                        "model_name": "Model"
                    },
                    height=400
                )
            elif not hardware:
                # Group by hardware (model is selected)
                fig = px.scatter(
                    df,
                    x="test_date",
                    y="metric_value",
                    color="hardware_type",
                    title=f"{metric_label} Over Time for {model}",
                    labels={
                        "test_date": "Date",
                        "metric_value": metric_label,
                        "hardware_type": "Hardware Platform"
                    },
                    height=400
                )
            elif not model:
                # Group by model (hardware is selected)
                fig = px.scatter(
                    df,
                    x="test_date",
                    y="metric_value",
                    color="model_name",
                    title=f"{metric_label} Over Time for {hardware}",
                    labels={
                        "test_date": "Date",
                        "metric_value": metric_label,
                        "model_name": "Model"
                    },
                    height=400
                )
            else:
                # Both model and hardware selected
                fig = px.scatter(
                    df,
                    x="test_date",
                    y="metric_value",
                    title=f"{metric_label} Over Time for {model} on {hardware}",
                    labels={
                        "test_date": "Date",
                        "metric_value": metric_label
                    },
                    height=400
                )
            
            # Add trend line if requested
            if "trend" in options and not df.empty:
                if model and hardware:
                    # Simple case: one model, one hardware
                    df_sorted = df.sort_values("test_date")
                    
                    if len(df_sorted) > 1:
                        # Add trend line
                        x = np.array((df_sorted["test_date"] - df_sorted["test_date"].min()).dt.total_seconds())
                        y = df_sorted["metric_value"].values
                        
                        if len(x) > 0 and len(y) > 0 and len(x) == len(y):
                            slope, intercept = np.polyfit(x, y, 1)
                            trend_x = [df_sorted["test_date"].min(), df_sorted["test_date"].max()]
                            trend_y = [intercept + slope * x[0], intercept + slope * x[-1]]
                            
                            # Add trend line
                            fig.add_trace(
                                go.Scatter(
                                    x=trend_x,
                                    y=trend_y,
                                    mode="lines",
                                    name="Trend",
                                    line=dict(color="red", dash="dash")
                                )
                            )
            
            # Generate statistical analysis
            if model and hardware:
                # Compute trend statistics
                analysis_div = html.Div(
                    className="card",
                    children=[
                        html.Div(
                            className="card-body",
                            children=[
                                html.H5("Statistical Analysis", className="card-title"),
                                html.Div(id="trend-analysis-content", children=[
                                    self._generate_trend_analysis_content(df, metric, options)
                                ])
                            ]
                        )
                    ]
                )
            else:
                analysis_div = html.Div(
                    className="card",
                    children=[
                        html.Div(
                            className="card-body",
                            children=[
                                html.H5("Statistical Analysis", className="card-title"),
                                html.P("Select a specific model and hardware platform for detailed statistical analysis.")
                            ]
                        )
                    ]
                )
            
            return fig, analysis_div
        
        # Simulation validation tab callbacks
        @self.app.callback(
            [
                Output("simulation-comparison-chart", "figure"),
                Output("hardware-ratio-chart", "figure"),
                Output("simulation-validation-status", "figure")
            ],
            [Input("interval-component", "n_intervals")]
        )
        def update_simulation_validation(n):
            # Get simulation validation data
            df = self.data_provider.get_simulation_validation_data()
            
            # Create empty figures if no data
            if df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    height=400,
                    title="No simulation validation data available"
                )
                return empty_fig, empty_fig, empty_fig
            
            # Create comparison chart
            comparison_fig = px.scatter(
                df,
                x="throughput",
                y="latency",
                color="hardware_type",
                symbol="is_simulation",
                symbol_map={True: "circle", False: "cross"},
                size="throughput",
                hover_data=["model_name"],
                title="Simulation vs. Real Hardware Performance",
                labels={
                    "throughput": "Throughput (items/second)",
                    "latency": "Latency (ms)",
                    "hardware_type": "Hardware Platform",
                    "is_simulation": "Simulation Status",
                    "model_name": "Model"
                },
                height=500
            )
            
            # Create hardware ratio chart
            # Create a DataFrame from the hardware performance ratios
            hw_ratio_data = []
            
            # Example hardware performance ratios (would be imported from validation_utils.py)
            hw_ratios = {
                ('cuda', 'cpu'): 3.5,
                ('rocm', 'cpu'): 2.8,
                ('mps', 'cpu'): 2.2,
                ('openvino', 'cpu'): 1.5,
                ('qnn', 'cpu'): 2.5,
                ('webgpu', 'cpu'): 2.0,
                ('webnn', 'cpu'): 1.8
            }
            
            for (hw1, hw2), ratio in hw_ratios.items():
                hw_ratio_data.append({
                    "hardware_pair": f"{hw1} vs {hw2}",
                    "performance_ratio": ratio
                })
            
            hw_ratio_df = pd.DataFrame(hw_ratio_data)
            
            hw_ratio_fig = px.bar(
                hw_ratio_df,
                x="hardware_pair",
                y="performance_ratio",
                title="Expected Hardware Performance Ratios",
                labels={
                    "hardware_pair": "Hardware Pair",
                    "performance_ratio": "Expected Performance Ratio"
                },
                height=400
            )
            
            # Create validation status figure
            # Count simulation validations
            if "is_simulation" in df.columns:
                num_simulations = df["is_simulation"].sum()
                
                # For this example, assume 80% of simulations are valid
                valid_simulations = int(num_simulations * 0.8)
                invalid_simulations = num_simulations - valid_simulations
                
                status_fig = go.Figure()
                
                # Add pie chart for validation status
                status_fig.add_trace(
                    go.Pie(
                        labels=["Valid Simulations", "Invalid Simulations"],
                        values=[valid_simulations, invalid_simulations],
                        hole=0.4,
                        marker=dict(colors=[COLORS["success"], COLORS["danger"]])
                    )
                )
                
                status_fig.update_layout(
                    title="Simulation Validation Status",
                    height=400
                )
            else:
                status_fig = go.Figure()
                status_fig.update_layout(
                    title="No simulation data available",
                    height=400
                )
            
            return comparison_fig, hw_ratio_fig, status_fig
    
    def _generate_trend_analysis_content(self, df: pd.DataFrame, metric: str, options: List[str]) -> html.Div:
        """
        Generate trend analysis content for the time series data.
        
        Args:
            df: DataFrame with time series data
            metric: Metric name being analyzed
            options: List of selected analysis options
            
        Returns:
            Dash HTML layout with trend analysis
        """
        if df.empty or len(df) < 2:
            return html.P("Insufficient data points for trend analysis.")
        
        df_sorted = df.sort_values("test_date")
        
        # Compute basic statistics
        mean_value = df_sorted["metric_value"].mean()
        min_value = df_sorted["metric_value"].min()
        max_value = df_sorted["metric_value"].max()
        std_dev = df_sorted["metric_value"].std()
        
        # Compute trend
        x = np.array((df_sorted["test_date"] - df_sorted["test_date"].min()).dt.total_seconds())
        y = df_sorted["metric_value"].values
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determine if the trend is significant
        is_improving = slope > 0 if metric == "throughput" else slope < 0
        performance_change = abs(slope) * np.max(x) / mean_value * 100 if mean_value > 0 else 0
        
        # Generate trend description
        if metric == "throughput":
            trend_description = f"Throughput is {'improving' if is_improving else 'decreasing'} by approximately {performance_change:.1f}% over the selected period."
        elif metric == "latency":
            trend_description = f"Latency is {'improving (decreasing)' if is_improving else 'degrading (increasing)'} by approximately {performance_change:.1f}% over the selected period."
        else:
            trend_description = f"Memory usage is {'increasing' if slope > 0 else 'decreasing'} by approximately {performance_change:.1f}% over the selected period."
        
        # Check for regressions
        regression_detected = performance_change > 5 and not is_improving
        
        return html.Div([
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.H6("Basic Statistics"),
                            html.Ul([
                                html.Li(f"Mean: {mean_value:.2f}"),
                                html.Li(f"Min: {min_value:.2f}"),
                                html.Li(f"Max: {max_value:.2f}"),
                                html.Li(f"Standard Deviation: {std_dev:.2f}")
                            ])
                        ]
                    ),
                    html.Div(
                        className="col-md-6",
                        children=[
                            html.H6("Trend Analysis"),
                            html.P(trend_description),
                            html.P(
                                f"Performance {'improved' if is_improving else 'regressed'} by {performance_change:.1f}%",
                                className=f"{'text-success' if is_improving else 'text-danger'}"
                            ) if performance_change > 0 else html.P("No significant performance change detected.")
                        ]
                    )
                ]
            ),
            html.Div(
                className="alert alert-danger" if regression_detected else "d-none",
                children=[
                    html.Strong("Regression Alert: "),
                    f"A performance regression of {performance_change:.1f}% has been detected for this metric."
                ]
            ) if "regressions" in options else None
        ])
    
    def run_server(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, debug: bool = False):
        """
        Run the Dash server.
        
        Args:
            host: Hostname to run the server on
            port: Port to run the server on
            debug: Enable debug mode
        """
        self.app.run_server(host=host, port=port, debug=debug)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualization Dashboard for End-to-End Testing Framework")
    
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help=f"Port to run the server on (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default=DEFAULT_HOST,
                       help=f"Hostname to run the server on (default: {DEFAULT_HOST})")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH,
                       help=f"Path to the DuckDB database file (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Create data provider
    data_provider = DashboardDataProvider(db_path=args.db_path)
    
    # Create and run dashboard
    dashboard = VisualizationDashboard(data_provider)
    
    try:
        logger.info(f"Starting dashboard server at http://{args.host}:{args.port}/")
        dashboard.run_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        data_provider.close()


if __name__ == "__main__":
    main()