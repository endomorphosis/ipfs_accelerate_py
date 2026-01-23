#!/usr/bin/env python3
"""
Enhanced Visualization Dashboard for Distributed Testing Framework

This module implements an advanced visualization dashboard for the distributed testing framework.
It provides real-time monitoring, interactive visualizations, and comprehensive performance metrics.

Features:
- Real-time performance monitoring
- Interactive data visualizations
- Comparative analysis across dimensions
- Regression detection and highlighting
- Time-series performance tracking
- Dynamic resource management visualization
- Multi-dimensional performance analysis
- Customizable dashboard layouts
- Export capabilities for reports and presentations
"""

import os
import sys
import json
import logging
import anyio
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("enhanced_visualization_dashboard")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import optional dependencies with graceful fallbacks
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    logger.warning("aiohttp not available. WebSocket features will be limited.")
    AIOHTTP_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Data manipulation features will be limited.")
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Interactive visualization features will be limited.")
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    logger.warning("Dash not available. Web dashboard features will be limited.")
    DASH_AVAILABLE = False

# Try to import visualization engine
try:
    from .visualization import VisualizationEngine
    VISUALIZATION_ENGINE_AVAILABLE = True
except ImportError:
    logger.warning("VisualizationEngine not available. Some features will be limited.")
    VISUALIZATION_ENGINE_AVAILABLE = False

# Try to import regression detection
try:
    from .regression_detection import RegressionDetector
    REGRESSION_DETECTOR_AVAILABLE = True
except ImportError:
    logger.warning("RegressionDetector not available. Regression detection features will be limited.")
    REGRESSION_DETECTOR_AVAILABLE = False

# Try to import regression visualization
try:
    from .regression_visualization import RegressionVisualization
    REGRESSION_VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("RegressionVisualization not available. Enhanced regression visualization features will be limited.")
    REGRESSION_VISUALIZATION_AVAILABLE = False

# Try to import DuckDB connector
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available. Database features will be limited.")
    DUCKDB_AVAILABLE = False

class EnhancedVisualizationDashboard:
    """Enhanced visualization dashboard for distributed testing framework."""
    
    def __init__(self, 
                result_aggregator=None, 
                db_path: str = "benchmark_db.duckdb",
                output_dir: str = "./visualizations/dashboard",
                host: str = "localhost",
                port: int = 8082,
                debug: bool = False,
                theme: str = "dark",
                enable_regression_detection: bool = True,
                enhanced_visualization: bool = True):
        """Initialize the enhanced visualization dashboard.
        
        Args:
            result_aggregator: Result aggregator for accessing result data
            db_path: Path to DuckDB database
            output_dir: Directory to save visualizations
            host: Host to bind the server to
            port: Port to bind the server to
            debug: Whether to enable debug mode
            theme: Dashboard theme (light or dark)
            enable_regression_detection: Whether to enable regression detection
            enhanced_visualization: Whether to use enhanced visualization features
        """
        self.result_aggregator = result_aggregator
        self.db_path = db_path
        self.output_dir = output_dir
        self.host = host
        self.port = port
        self.debug = debug
        self.theme = theme
        self.enable_regression_detection = enable_regression_detection
        self.enhanced_visualization = enhanced_visualization
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create regression visualizations subdirectory
        self.regression_output_dir = os.path.join(output_dir, "regression")
        os.makedirs(self.regression_output_dir, exist_ok=True)
        
        # Initialize database connection if available
        self.db_conn = None
        if DUCKDB_AVAILABLE:
            try:
                self.db_conn = duckdb.connect(db_path)
                logger.info(f"Connected to DuckDB database: {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to DuckDB database: {e}")
        
        # Initialize visualization engine if available
        self.visualization_engine = None
        if VISUALIZATION_ENGINE_AVAILABLE:
            try:
                self.visualization_engine = VisualizationEngine(
                    result_aggregator=result_aggregator,
                    output_dir=output_dir
                )
                logger.info("Visualization engine initialized")
            except Exception as e:
                logger.error(f"Error initializing visualization engine: {e}")
        
        # Initialize regression detector if available and enabled
        self.regression_detector = None
        if REGRESSION_DETECTOR_AVAILABLE and self.enable_regression_detection:
            try:
                self.regression_detector = RegressionDetector(
                    db_conn=self.db_conn
                )
                logger.info("Regression detector initialized")
            except Exception as e:
                logger.error(f"Error initializing regression detector: {e}")
        
        # Initialize regression visualization if available and enhanced viz is enabled
        self.regression_visualization = None
        if REGRESSION_VISUALIZATION_AVAILABLE and self.enhanced_visualization:
            try:
                self.regression_visualization = RegressionVisualization(
                    output_dir=self.regression_output_dir
                )
                # Set the theme to match the dashboard
                self.regression_visualization.set_theme(self.theme)
                logger.info("Regression visualization initialized")
            except Exception as e:
                logger.error(f"Error initializing regression visualization: {e}")
        
        # Initialize Dash app if available
        self.app = None
        if DASH_AVAILABLE:
            try:
                self.app = dash.Dash(
                    __name__,
                    external_stylesheets=[
                        dbc.themes.DARKLY if theme == "dark" else dbc.themes.BOOTSTRAP,
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
                    ],
                    suppress_callback_exceptions=True
                )
                
                # Set up app layout and callbacks
                self._setup_app_layout()
                self._setup_app_callbacks()
                logger.info("Dash app initialized")
            except Exception as e:
                logger.error(f"Error initializing Dash app: {e}")
        
        # Data cache for dashboard
        self.data_cache = {
            "performance_metrics": {},
            "hardware_stats": {},
            "test_results": {},
            "worker_status": {},
            "resource_usage": {},
            "regression_analysis": {
                "regressions_by_metric": {},
                "regression_report": None,
                "correlation_analysis": None,
                "regression_visualizations": {},
                "regression_heatmap": None,
                "visualization_options": {
                    "include_confidence_intervals": True,
                    "include_trend_lines": True,
                    "include_annotations": True,
                    "export_format": "html"
                },
                "enhanced_results": {}
            },
            "last_updated": datetime.datetime.now()
        }
        
        # Dashboard configuration
        self.config = {
            "refresh_interval": 5,  # seconds
            "max_history_points": 100,
            "chart_height": 400,
            "chart_width": 600,
            "show_animations": True,
            "enable_real_time_updates": True,
            "default_metrics": ["throughput", "latency", "memory_usage"],
            "default_dimensions": ["hardware", "model", "precision", "batch_size"],
            "color_schemes": {
                "light": {
                    "background": "#ffffff",
                    "text": "#333333",
                    "primary": "#007bff",
                    "secondary": "#6c757d",
                    "success": "#28a745",
                    "warning": "#ffc107",
                    "danger": "#dc3545",
                    "info": "#17a2b8"
                },
                "dark": {
                    "background": "#222222",
                    "text": "#f8f9fa",
                    "primary": "#375a7f",
                    "secondary": "#444444",
                    "success": "#00bc8c",
                    "warning": "#f39c12",
                    "danger": "#e74c3c",
                    "info": "#3498db"
                }
            }
        }
        
        # WebSocket server and clients
        self.websocket_server = None
        self.websocket_clients = set()
        
        logger.info("Enhanced visualization dashboard initialized")
    
    def _setup_app_layout(self):
        """Set up the Dash app layout."""
        if not DASH_AVAILABLE or self.app is None:
            return
        
        # Get color scheme based on theme
        colors = self.config["color_schemes"][self.theme]
        
        # Create app layout
        self.app.layout = html.Div([
            # Navigation bar
            dbc.Navbar(
                dbc.Container([
                    dbc.Row([
                        dbc.Col(
                            html.A(
                                # Logo and title
                                dbc.Row([
                                    dbc.Col(html.I(className="fas fa-chart-line me-2")),
                                    dbc.Col(dbc.NavbarBrand("Distributed Testing Dashboard", className="ms-2")),
                                ],
                                align="center",
                                className="g-0",
                                ),
                                href="#",
                                style={"textDecoration": "none"},
                            )
                        ),
                        # Navigation links
                        dbc.Col(
                            dbc.Nav([
                                dbc.NavItem(dbc.NavLink("Overview", active=True, href="#overview")),
                                dbc.NavItem(dbc.NavLink("Performance", href="#performance")),
                                dbc.NavItem(dbc.NavLink("Resources", href="#resources")),
                                dbc.NavItem(dbc.NavLink("Tests", href="#tests")),
                                dbc.NavItem(dbc.NavLink("Analysis", href="#analysis")),
                                dbc.NavItem(dbc.NavLink("Settings", href="#settings")),
                            ],
                            pills=True,
                            className="me-auto",
                            ),
                            width="auto",
                        ),
                        # Real-time status indicator
                        dbc.Col(
                            html.Div([
                                html.Span("Status: ", className="me-2"),
                                html.Span("Active", id="status-indicator", className="text-success"),
                                html.Span(" • ", className="mx-2"),
                                html.Span("Last updated: ", className="me-1"),
                                html.Span(datetime.datetime.now().strftime("%H:%M:%S"), id="last-updated"),
                            ],
                            className="d-flex align-items-center"),
                            width="auto",
                        ),
                    ],
                    align="center",
                    ),
                ],
                fluid=True,
                ),
                color=colors["primary"],
                dark=True,
                className="mb-4",
            ),
            
            # Main content
            dbc.Container([
                # Overview section
                html.Div([
                    html.H2("System Overview", id="overview", className="mb-4"),
                    
                    # Stats cards row
                    dbc.Row([
                        # Workers card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Active Workers", className="card-title text-center"),
                                    html.H2("0", id="active-workers-count", className="text-center display-4"),
                                    html.P("Total workers in the system", className="text-center text-muted small"),
                                ])
                            ],
                            className="shadow-sm"),
                            width=3,
                        ),
                        # Tests card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Running Tests", className="card-title text-center"),
                                    html.H2("0", id="running-tests-count", className="text-center display-4"),
                                    html.P("Currently executing tests", className="text-center text-muted small"),
                                ])
                            ],
                            className="shadow-sm"),
                            width=3,
                        ),
                        # Resource usage card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Resource Usage", className="card-title text-center"),
                                    html.H2("0%", id="resource-usage-percent", className="text-center display-4"),
                                    html.P("Average across all workers", className="text-center text-muted small"),
                                ])
                            ],
                            className="shadow-sm"),
                            width=3,
                        ),
                        # Performance card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Task Throughput", className="card-title text-center"),
                                    html.H2("0", id="task-throughput", className="text-center display-4"),
                                    html.P("Tasks per minute", className="text-center text-muted small"),
                                ])
                            ],
                            className="shadow-sm"),
                            width=3,
                        ),
                    ],
                    className="mb-4"),
                    
                    # System health chart
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("System Health Metrics"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="system-health-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "300px"},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=8,
                        ),
                        
                        # Status cards
                        dbc.Col([
                            # Cluster status card
                            dbc.Card([
                                dbc.CardHeader("Cluster Status"),
                                dbc.CardBody([
                                    html.Div([
                                        html.Span("Leader Node: ", className="fw-bold"),
                                        html.Span("worker-001", id="leader-node"),
                                    ],
                                    className="mb-2"),
                                    html.Div([
                                        html.Span("Coordinator Health: ", className="fw-bold"),
                                        html.Span("Healthy", id="coordinator-health", className="text-success"),
                                    ],
                                    className="mb-2"),
                                    html.Div([
                                        html.Span("Nodes: ", className="fw-bold"),
                                        html.Span("5 online, 0 offline", id="node-status"),
                                    ],
                                    className="mb-2"),
                                    html.Div([
                                        html.Span("Last Failover: ", className="fw-bold"),
                                        html.Span("None", id="last-failover"),
                                    ]),
                                ])
                            ],
                            className="shadow-sm mb-3"),
                            
                            # Alert status card
                            dbc.Card([
                                dbc.CardHeader("Alerts"),
                                dbc.CardBody([
                                    html.Div(id="alerts-container", style={"maxHeight": "130px", "overflowY": "auto"})
                                ])
                            ],
                            className="shadow-sm"),
                            
                        ],
                        width=4),
                    ],
                    className="mb-4"),
                ],
                className="mb-5"),
                
                # Performance section
                html.Div([
                    html.H2("Performance Metrics", id="performance", className="mb-4"),
                    
                    # Performance filter controls
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Filters", className="card-title mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Metrics", className="form-label"),
                                            dcc.Dropdown(
                                                id="metrics-dropdown",
                                                options=[
                                                    {"label": "Throughput", "value": "throughput"},
                                                    {"label": "Latency", "value": "latency"},
                                                    {"label": "Memory Usage", "value": "memory_usage"},
                                                    {"label": "CPU Usage", "value": "cpu_usage"},
                                                    {"label": "GPU Usage", "value": "gpu_usage"}
                                                ],
                                                value=["throughput", "latency", "memory_usage"],
                                                multi=True,
                                                className="mb-3",
                                            ),
                                        ],
                                        width=4),
                                        dbc.Col([
                                            html.Label("Models", className="form-label"),
                                            dcc.Dropdown(
                                                id="models-dropdown",
                                                options=[
                                                    {"label": "All Models", "value": "all"},
                                                    {"label": "BERT", "value": "bert"},
                                                    {"label": "GPT-2", "value": "gpt2"},
                                                    {"label": "T5", "value": "t5"},
                                                    {"label": "ViT", "value": "vit"}
                                                ],
                                                value="all",
                                                className="mb-3",
                                            ),
                                        ],
                                        width=4),
                                        dbc.Col([
                                            html.Label("Hardware", className="form-label"),
                                            dcc.Dropdown(
                                                id="hardware-dropdown",
                                                options=[
                                                    {"label": "All Hardware", "value": "all"},
                                                    {"label": "CPU", "value": "cpu"},
                                                    {"label": "CUDA", "value": "cuda"},
                                                    {"label": "ROCm", "value": "rocm"},
                                                    {"label": "WebGPU", "value": "webgpu"},
                                                    {"label": "WebNN", "value": "webnn"}
                                                ],
                                                value="all",
                                                className="mb-3",
                                            ),
                                        ],
                                        width=4),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Time Range", className="form-label"),
                                            dcc.RadioItems(
                                                id="time-range-radio",
                                                options=[
                                                    {"label": "Last Hour", "value": "1h"},
                                                    {"label": "Last Day", "value": "1d"},
                                                    {"label": "Last Week", "value": "7d"},
                                                    {"label": "Last Month", "value": "30d"},
                                                ],
                                                value="1h",
                                                inline=True,
                                                className="mb-3",
                                            ),
                                        ],
                                        width=8),
                                        dbc.Col([
                                            html.Label("Refresh Rate", className="form-label"),
                                            dcc.Dropdown(
                                                id="refresh-rate-dropdown",
                                                options=[
                                                    {"label": "5 seconds", "value": 5},
                                                    {"label": "10 seconds", "value": 10},
                                                    {"label": "30 seconds", "value": 30},
                                                    {"label": "1 minute", "value": 60},
                                                    {"label": "Off", "value": 0},
                                                ],
                                                value=5,
                                                className="mb-3",
                                            ),
                                        ],
                                        width=2),
                                        dbc.Col([
                                            html.Label("Theme", className="form-label"),
                                            dcc.Dropdown(
                                                id="theme-dropdown",
                                                options=[
                                                    {"label": "Dark", "value": "dark"},
                                                    {"label": "Light", "value": "light"},
                                                ],
                                                value=self.theme,
                                                className="mb-3",
                                            ),
                                        ],
                                        width=2),
                                    ]),
                                    dbc.Button("Apply Filters", id="apply-filters-btn", color="primary", className="w-100"),
                                ])
                            ],
                            className="shadow-sm"),
                            width=12,
                        ),
                    ],
                    className="mb-4"),
                    
                    # Performance charts
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Performance Over Time"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="performance-time-chart",
                                        config={"displayModeBar": True},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=12,
                        ),
                    ],
                    className="mb-4"),
                    
                    # Performance comparison and breakdown
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Performance by Hardware"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="hardware-comparison-chart",
                                        config={"displayModeBar": True},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Performance by Model"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="model-comparison-chart",
                                        config={"displayModeBar": True},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=6,
                        ),
                    ],
                    className="mb-4"),
                ],
                className="mb-5"),
                
                # Resources section
                html.Div([
                    html.H2("Resource Management", id="resources", className="mb-4"),
                    
                    # Resource allocation charts
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Resource Allocation"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="resource-allocation-chart",
                                        config={"displayModeBar": True},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Worker Utilization"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="worker-utilization-chart",
                                        config={"displayModeBar": True},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=6,
                        ),
                    ],
                    className="mb-4"),
                    
                    # Resource scaling and prediction
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Dynamic Scaling Events"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="scaling-events-chart",
                                        config={"displayModeBar": True},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Resource Prediction"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="resource-prediction-chart",
                                        config={"displayModeBar": True},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=6,
                        ),
                    ],
                    className="mb-4"),
                ],
                className="mb-5"),
                
                # Tests section
                html.Div([
                    html.H2("Tests Overview", id="tests", className="mb-4"),
                    
                    # Test summary and filters
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Test Execution Summary"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Span("Total Tests: ", className="fw-bold"),
                                                html.Span("0", id="total-tests"),
                                            ],
                                            className="mb-2"),
                                            html.Div([
                                                html.Span("Completed: ", className="fw-bold"),
                                                html.Span("0", id="completed-tests"),
                                            ],
                                            className="mb-2"),
                                            html.Div([
                                                html.Span("Running: ", className="fw-bold"),
                                                html.Span("0", id="running-tests"),
                                            ],
                                            className="mb-2"),
                                            html.Div([
                                                html.Span("Failed: ", className="fw-bold"),
                                                html.Span("0", id="failed-tests"),
                                            ]),
                                        ],
                                        width=6),
                                        dbc.Col([
                                            html.Div([
                                                html.Span("Success Rate: ", className="fw-bold"),
                                                html.Span("0%", id="success-rate"),
                                            ],
                                            className="mb-2"),
                                            html.Div([
                                                html.Span("Avg. Duration: ", className="fw-bold"),
                                                html.Span("0s", id="avg-duration"),
                                            ],
                                            className="mb-2"),
                                            html.Div([
                                                html.Span("Last Test: ", className="fw-bold"),
                                                html.Span("N/A", id="last-test"),
                                            ]),
                                        ],
                                        width=6),
                                    ]),
                                ])
                            ],
                            className="shadow-sm"),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Test Execution Status"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="test-status-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "150px"},
                                    )
                                ])
                            ],
                            className="shadow-sm"),
                            width=8,
                        ),
                    ],
                    className="mb-4"),
                    
                    # Test results table
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader([
                                    html.Div([
                                        html.Span("Recent Test Results"),
                                        dbc.Button(
                                            html.I(className="fas fa-sync-alt"),
                                            id="refresh-tests-btn",
                                            color="link",
                                            size="sm",
                                            className="float-end",
                                        ),
                                    ]),
                                ]),
                                dbc.CardBody([
                                    html.Div(
                                        id="test-results-table-container",
                                        style={"height": "400px", "overflowY": "auto"}
                                    ),
                                ])
                            ],
                            className="shadow-sm"),
                            width=12,
                        ),
                    ],
                    className="mb-4"),
                ],
                className="mb-5"),
                
                # Analysis section
                html.Div([
                    html.H2("Performance Analysis", id="analysis", className="mb-4"),
                    
                    # Analysis tabs
                    dbc.Tabs([
                        # Trend analysis tab
                        dbc.Tab([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Performance Trends", className="mb-3"),
                                            dcc.Graph(
                                                id="performance-trend-chart",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=8),
                                        dbc.Col([
                                            html.H5("Trend Analysis", className="mb-3"),
                                            html.Div(id="trend-analysis-content", className="p-3 border rounded", style={"height": "400px", "overflowY": "auto"}),
                                        ],
                                        width=4),
                                    ]),
                                ])
                            ],
                            className="shadow-sm mt-3"),
                        ],
                        label="Trend Analysis",
                        tab_id="tab-trend-analysis"),
                        
                        # Regression detection tab
                        dbc.Tab([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Performance Regressions", className="mb-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Label("Metric for Regression Analysis"),
                                                    dcc.Dropdown(
                                                        id="regression-metric-dropdown",
                                                        options=[
                                                            {"label": "Throughput", "value": "throughput_items_per_second"},
                                                            {"label": "Latency", "value": "latency_ms"},
                                                            {"label": "Memory Usage", "value": "memory_usage_mb"},
                                                            {"label": "CPU Usage", "value": "cpu_usage"},
                                                            {"label": "GPU Usage", "value": "gpu_usage"}
                                                        ],
                                                        value="latency_ms",
                                                        className="mb-3",
                                                    ),
                                                ],
                                                width=6),
                                                dbc.Col([
                                                    html.Label("Analysis Configuration"),
                                                    dbc.Button(
                                                        "Run Regression Analysis",
                                                        id="run-regression-analysis-btn",
                                                        color="primary",
                                                        className="w-100 mb-3",
                                                    ),
                                                ],
                                                width=6),
                                            ]),
                                            dcc.Graph(
                                                id="regression-chart",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=8),
                                        dbc.Col([
                                            html.H5("Regression Details", className="mb-3"),
                                            html.Div(id="regression-details-content", className="p-3 border rounded", style={"height": "400px", "overflowY": "auto"}),
                                        ],
                                        width=4),
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Regression Analysis Options", className="mb-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Button(
                                                        "Run Correlation Analysis",
                                                        id="run-correlation-analysis-btn",
                                                        color="secondary",
                                                        className="w-100 mb-3",
                                                    ),
                                                ], width=4),
                                                dbc.Col([
                                                    dbc.Button(
                                                        "Generate Regression Report",
                                                        id="generate-regression-report-btn",
                                                        color="info",
                                                        className="w-100 mb-3",
                                                    ),
                                                ], width=4),
                                                dbc.Col([
                                                    dbc.Button(
                                                        "Export Visualization",
                                                        id="export-regression-viz-btn",
                                                        color="primary",
                                                        className="w-100 mb-3",
                                                    ),
                                                ], width=4),
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.H6("Visualization Features", className="mb-2"),
                                                    dbc.Card(
                                                        dbc.CardBody([
                                                            dbc.FormGroup([
                                                                dbc.Label("Display Options"),
                                                                dbc.Checklist(
                                                                    options=[
                                                                        {"label": "Show Confidence Intervals", "value": "ci"},
                                                                        {"label": "Show Trend Lines", "value": "trend"},
                                                                        {"label": "Show Annotations", "value": "annotations"},
                                                                    ],
                                                                    value=["ci", "trend", "annotations"],
                                                                    id="regression-viz-options",
                                                                    inline=True,
                                                                    switch=True,
                                                                ),
                                                            ]),
                                                            html.Hr(className="my-2"),
                                                            dbc.FormGroup([
                                                                dbc.Label("Export Format"),
                                                                dcc.Dropdown(
                                                                    id="export-format-dropdown",
                                                                    options=[
                                                                        {"label": "HTML", "value": "html"},
                                                                        {"label": "PNG Image", "value": "png"},
                                                                        {"label": "SVG Image", "value": "svg"},
                                                                        {"label": "PDF Document", "value": "pdf"},
                                                                        {"label": "JSON Data", "value": "json"}
                                                                    ],
                                                                    value="html",
                                                                    clearable=False,
                                                                    className="mb-2",
                                                                ),
                                                            ]),
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Button(
                                                                        "Export Visualization",
                                                                        id="export-regression-viz-btn-inline",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="w-100",
                                                                    ),
                                                                ], width=6),
                                                                dbc.Col([
                                                                    dbc.Button(
                                                                        "Generate Report",
                                                                        id="generate-regression-report-btn-inline",
                                                                        color="info",
                                                                        size="sm",
                                                                        className="w-100",
                                                                    ),
                                                                ], width=6),
                                                            ]),
                                                            html.Div(id="export-status-inline", className="mt-2 small text-muted"),
                                                        ]),
                                                        className="mb-3",
                                                    ),
                                                ], width=12),
                                            ]),
                                        ], width=12),
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Metric Correlations", className="mb-3"),
                                            dcc.Graph(
                                                id="correlation-chart",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=12),
                                    ]),
                                    # Notification section for exports and reports
                                    html.Div([
                                        html.H5("Export and Report Status", className="mb-3"),
                                        html.Div(id="regression-export-info", className="alert alert-info", style={"display": "none"}),
                                        
                                        # Export and report status
                                        dbc.Row([
                                            dbc.Col([
                                                html.Div(id="export-status", className="alert alert-info p-2", style={"fontSize": "0.9rem"}),
                                            ], width=6),
                                            dbc.Col([
                                                html.Div(id="report-status", className="alert alert-info p-2", style={"fontSize": "0.9rem"}),
                                            ], width=6),
                                        ]),
                                    ], className="mt-4"),
                                ])
                            ],
                            className="shadow-sm mt-3"),
                        ],
                        label="Regression Detection",
                        tab_id="tab-regression-detection"),
                        
                        # Correlation analysis tab
                        dbc.Tab([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Metric Correlations", className="mb-3"),
                                            dcc.Graph(
                                                id="correlation-chart",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=12),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Correlation Controls", className="mb-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Label("X-Axis Metric"),
                                                    dcc.Dropdown(
                                                        id="correlation-x-dropdown",
                                                        options=[
                                                            {"label": "Throughput", "value": "throughput"},
                                                            {"label": "Latency", "value": "latency"},
                                                            {"label": "Memory Usage", "value": "memory_usage"},
                                                            {"label": "CPU Usage", "value": "cpu_usage"},
                                                            {"label": "GPU Usage", "value": "gpu_usage"}
                                                        ],
                                                        value="latency",
                                                    ),
                                                ],
                                                width=6),
                                                dbc.Col([
                                                    html.Label("Y-Axis Metric"),
                                                    dcc.Dropdown(
                                                        id="correlation-y-dropdown",
                                                        options=[
                                                            {"label": "Throughput", "value": "throughput"},
                                                            {"label": "Latency", "value": "latency"},
                                                            {"label": "Memory Usage", "value": "memory_usage"},
                                                            {"label": "CPU Usage", "value": "cpu_usage"},
                                                            {"label": "GPU Usage", "value": "gpu_usage"}
                                                        ],
                                                        value="throughput",
                                                    ),
                                                ],
                                                width=6),
                                            ],
                                            className="mb-3"),
                                            dbc.Button("Update Correlation", id="update-correlation-btn", color="primary", className="w-100"),
                                        ],
                                        width=12),
                                    ]),
                                ])
                            ],
                            className="shadow-sm mt-3"),
                        ],
                        label="Correlation Analysis",
                        tab_id="tab-correlation-analysis"),
                        
                        # Anomaly detection tab
                        dbc.Tab([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Anomaly Detection", className="mb-3"),
                                            dcc.Graph(
                                                id="anomaly-chart",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=8),
                                        dbc.Col([
                                            html.H5("Detected Anomalies", className="mb-3"),
                                            html.Div(id="anomaly-details-content", className="p-3 border rounded", style={"height": "400px", "overflowY": "auto"}),
                                        ],
                                        width=4),
                                    ]),
                                ])
                            ],
                            className="shadow-sm mt-3"),
                        ],
                        label="Anomaly Detection",
                        tab_id="tab-anomaly-detection"),
                    ],
                    id="analysis-tabs",
                    active_tab="tab-trend-analysis"),
                ],
                className="mb-5"),
                
                # Footer
                html.Footer([
                    html.Hr(),
                    html.P([
                        "Distributed Testing Framework Dashboard • ",
                        html.Span(datetime.datetime.now().year),
                        " • ",
                        html.A("Documentation", href="#", className="text-decoration-none"),
                    ],
                    className="text-center text-muted small"),
                ]),
                
                # Hidden divs for storing data
                html.Div(id="hidden-data-store", style={"display": "none"}),
                dcc.Store(id="dashboard-data-store"),
                dcc.Interval(id="refresh-interval", interval=5*1000, n_intervals=0),  # 5 seconds refresh
            ],
            fluid=True),
        ])
    
    def _setup_app_callbacks(self):
        """Set up the Dash app callbacks."""
        if not DASH_AVAILABLE or self.app is None:
            return
        
        # Callback for refreshing data
        @self.app.callback(
            [Output("dashboard-data-store", "data"),
             Output("last-updated", "children")],
            [Input("refresh-interval", "n_intervals"),
             Input("apply-filters-btn", "n_clicks"),
             Input("refresh-tests-btn", "n_clicks")],
            [State("metrics-dropdown", "value"),
             State("models-dropdown", "value"),
             State("hardware-dropdown", "value"),
             State("time-range-radio", "value"),
             State("refresh-rate-dropdown", "value"),
             State("theme-dropdown", "value")],
            prevent_initial_call=False
        )
        def refresh_dashboard_data(n_intervals, filter_clicks, refresh_tests_clicks, 
                                metrics, models, hardware, time_range, refresh_rate, theme):
            """Refresh dashboard data."""
            # Update refresh interval if needed
            if refresh_rate is not None and dash.callback_context.triggered_id == "apply-filters-btn":
                interval_component = dcc.Interval(id="refresh-interval", interval=refresh_rate*1000, n_intervals=0)
                self.app.callback_context.response.set({"refreshInterval": refresh_rate*1000})
            
            # Update theme if changed
            if theme is not None and theme != self.theme and dash.callback_context.triggered_id == "apply-filters-btn":
                self.theme = theme
                # Update regression visualization theme if available
                if self.regression_visualization:
                    self.regression_visualization.set_theme(theme)
                logger.info(f"Dashboard theme changed to {theme}")
            
            # Get updated data (mocked for now)
            dashboard_data = self._get_dashboard_data(metrics, models, hardware, time_range)
            
            # Update last updated timestamp
            last_updated = datetime.datetime.now().strftime("%H:%M:%S")
            
            return dashboard_data, last_updated
        
        # Callback for running regression analysis
        @self.app.callback(
            [Output("regression-chart", "figure"),
             Output("regression-details-content", "children"),
             Output("regression-viz-options", "value", allow_duplicate=True)],
            [Input("run-regression-analysis-btn", "n_clicks"),
             Input("dashboard-data-store", "data"),
             Input("regression-viz-options", "value")],
            [State("regression-metric-dropdown", "value"),
             State("export-format-dropdown", "value")],
            prevent_initial_call=True
        )
        def run_regression_analysis(n_clicks, data, viz_options, selected_metric, export_format):
            """Run regression analysis on selected metric with visualization options."""
            if not data or not self.regression_detector or selected_metric is None:
                # Return empty figure if no data or regression detector
                return go.Figure().update_layout(
                    title="No regression data available",
                    template="plotly_dark" if self.theme == "dark" else "plotly"
                ), "No regression data available", ["ci", "trend", "annotations"]
                
            # Update visualization options based on UI selections
            include_confidence_intervals = "ci" in viz_options if viz_options else True
            include_trend_lines = "trend" in viz_options if viz_options else True
            include_annotations = "annotations" in viz_options if viz_options else True
            
            # Update data cache with current visualization options
            viz_options_dict = {
                "include_confidence_intervals": include_confidence_intervals,
                "include_trend_lines": include_trend_lines,
                "include_annotations": include_annotations,
                "export_format": export_format if export_format else "html"
            }
            self.data_cache["regression_analysis"]["visualization_options"].update(viz_options_dict)
            
            # Log the visualization options being used
            logger.info(f"Using visualization options: {viz_options_dict}")
            
            # Check if triggered by button click or data update
            is_button_click = dash.callback_context.triggered_id == "run-regression-analysis-btn"
            
            # Get performance metrics data
            performance = data.get("performance_metrics", {})
            
            # Find the metric data
            time_series_data = {}
            regressions_by_metric = {}
            
            if selected_metric in performance:
                # For each hardware series
                for hw, series_data in performance[selected_metric].items():
                    if "timestamps" in series_data and "values" in series_data:
                        time_series_key = f"{selected_metric}_{hw}"
                        time_series_data[time_series_key] = series_data
                        
                        # Run regression detection if triggered by button click
                        if is_button_click and self.regression_detector:
                            try:
                                regressions = self.regression_detector.detect_regressions(series_data, selected_metric)
                                if regressions:
                                    regressions_by_metric[time_series_key] = regressions
                            except Exception as e:
                                logger.error(f"Error detecting regressions: {e}")
            
            # Store regression results in data cache
            if regressions_by_metric:
                self.data_cache["regression_analysis"]["regressions_by_metric"].update(regressions_by_metric)
                
                # Generate regression report if we have data
                if time_series_data and self.regression_detector:
                    try:
                        report = self.regression_detector.generate_regression_report(time_series_data, regressions_by_metric)
                        self.data_cache["regression_analysis"]["regression_report"] = report
                    except Exception as e:
                        logger.error(f"Error generating regression report: {e}")
            
            # Create visualization for the first detected series
            fig = go.Figure().update_layout(
                title=f"No regression data available for {selected_metric}",
                template="plotly_dark" if self.theme == "dark" else "plotly"
            )
            
            # Get the first time series for visualization
            if time_series_data and self.regression_detector:
                first_key = list(time_series_data.keys())[0]
                first_data = time_series_data[first_key]
                
                # Get the regressions for this time series
                regressions = regressions_by_metric.get(first_key, [])
                
                # Create visualization - use enhanced visualization if available
                try:
                    # Get visualization options from data cache
                    viz_options = self.data_cache["regression_analysis"]["visualization_options"]
                    
                    if self.regression_visualization and self.enhanced_visualization:
                        # Get visualization options from the data cache
                        viz_options_dict = self.data_cache["regression_analysis"]["visualization_options"]
                        
                        # Use enhanced visualization with user-selected options
                        fig_dict = self.regression_visualization.create_interactive_regression_figure(
                            first_data, 
                            regressions, 
                            selected_metric,
                            title=f"Regression Analysis for {selected_metric}",
                            include_annotations=viz_options_dict.get("include_annotations", True),
                            include_confidence_intervals=viz_options_dict.get("include_confidence_intervals", True),
                            include_trend_lines=viz_options_dict.get("include_trend_lines", True)
                        )
                        
                        # Store the visualization in the data cache for export
                        if fig_dict:
                            self.data_cache["regression_analysis"]["enhanced_results"]["current_figure"] = fig_dict
                            logger.info(f"Created enhanced regression visualization with options: {viz_options_dict}")
                    else:
                        # Fall back to basic visualization from regression detector
                        fig_dict = self.regression_detector.create_regression_visualization(
                            first_data, 
                            regressions, 
                            selected_metric,
                            title=f"Regression Analysis for {selected_metric}"
                        )
                    
                    if fig_dict:
                        fig = go.Figure(fig_dict)
                except Exception as e:
                    logger.error(f"Error creating regression visualization: {e}")
            
            # Create HTML content for regression details
            details_content = []
            
            report = self.data_cache["regression_analysis"]["regression_report"]
            if report:
                # Add summary
                summary = report.get("summary", {})
                details_content.extend([
                    html.H6("Regression Summary"),
                    html.Div([
                        html.Strong("Total Metrics Analyzed: "),
                        html.Span(f"{summary.get('total_metrics_analyzed', 0)}"),
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Total Regressions: "),
                        html.Span(f"{summary.get('total_regressions_detected', 0)}"),
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Significant Regressions: "),
                        html.Span(f"{summary.get('significant_regressions', 0)}"),
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Critical: "),
                        html.Span(f"{summary.get('critical_regressions', 0)}", className="text-danger"),
                        html.Strong(" High: "),
                        html.Span(f"{summary.get('high_regressions', 0)}", className="text-warning"),
                        html.Strong(" Medium: "),
                        html.Span(f"{summary.get('medium_regressions', 0)}", className="text-info"),
                        html.Strong(" Low: "),
                        html.Span(f"{summary.get('low_regressions', 0)}", className="text-success"),
                    ], className="mb-3"),
                    
                    html.Hr(),
                ])
                
                # Add regression details
                if "regressions" in report and report["regressions"]:
                    details_content.append(html.H6("Regression Details"))
                    
                    for regression in report["regressions"]:
                        # Get color based on severity
                        severity = regression.get("severity", "none")
                        severity_class = {
                            "critical": "danger",
                            "high": "warning",
                            "medium": "info",
                            "low": "success",
                            "none": "secondary"
                        }.get(severity, "secondary")
                        
                        details_content.append(
                            dbc.Card([
                                dbc.CardHeader([
                                    html.Span(
                                        f"{regression.get('display_name', regression.get('metric', 'Unknown'))}",
                                        className="fw-bold me-2"
                                    ),
                                    dbc.Badge(
                                        f"{severity.title()}",
                                        color=severity_class,
                                        className="ms-1"
                                    ),
                                    html.Span(
                                        f" {regression.get('percentage_change', 0):.2f}% change",
                                        className="float-end"
                                    ),
                                ]),
                                dbc.CardBody([
                                    html.Div([
                                        html.Strong("Before: "),
                                        html.Span(f"{regression.get('before_mean', 0):.2f} {regression.get('unit', '')}"),
                                        html.Strong(" After: "),
                                        html.Span(f"{regression.get('after_mean', 0):.2f} {regression.get('unit', '')}"),
                                    ], className="mb-2"),
                                    html.Div([
                                        html.Strong("Statistical Significance: "),
                                        html.Span(
                                            f"{regression.get('significance', 0)*100:.1f}%",
                                            className=f"text-{'success' if regression.get('is_significant', False) else 'secondary'}"
                                        ),
                                    ], className="mb-2"),
                                    html.Div([
                                        html.Strong("Type: "),
                                        html.Span(
                                            f"{'Regression' if regression.get('is_regression', False) else 'Improvement'}",
                                            className=f"text-{'danger' if regression.get('is_regression', False) else 'success'}"
                                        ),
                                    ], className="mb-2"),
                                ]),
                            ], className="mb-3")
                        )
                    
                else:
                    details_content.append(html.P("No significant regressions detected."))
            else:
                details_content.append(html.P("No regression report available. Run regression analysis to generate report."))
            
            # Return viz_options to ensure checkbox state is consistent
            return fig, details_content, viz_options
        
        # Callback for exporting regression visualization (handles both export buttons)
        @self.app.callback(
            [Output("export-status", "children"),
             Output("export-status-inline", "children")],
            [Input("export-regression-viz-btn", "n_clicks"),
             Input("export-regression-viz-btn-inline", "n_clicks")],
            [State("export-format-dropdown", "value")],
            prevent_initial_call=True
        )
        def export_regression_visualization(n_clicks, n_clicks_inline, export_format):
            """Export the current regression visualization."""
            # Check which button was clicked
            trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
            
            if (not n_clicks and not n_clicks_inline) or not export_format or not self.regression_visualization:
                return "No visualization to export", ""
                
            # Get the current visualization from the data cache
            current_figure = self.data_cache["regression_analysis"]["enhanced_results"].get("current_figure")
            if not current_figure:
                return "No visualization available for export", "No visualization available"
                
            try:
                # Generate timestamp for filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"regression_viz_{timestamp}.{export_format}"
                output_path = os.path.join(self.regression_output_dir, filename)
                
                # Export the visualization
                exported_path = self.regression_visualization.export_regression_visualization(
                    current_figure,
                    output_path=output_path,
                    format=export_format
                )
                
                status_msg = f"Visualization exported to: {os.path.basename(exported_path)}" if exported_path else "Failed to export visualization"
                inline_msg = f"Exported: {os.path.basename(exported_path)}" if exported_path else "Export failed"
                
                return status_msg, inline_msg
            except Exception as e:
                logger.error(f"Error exporting visualization: {e}")
                error_msg = f"Error exporting visualization: {str(e)}"
                return error_msg, "Export error"
                
        # Callback for generating comprehensive regression report (handles both report buttons)
        @self.app.callback(
            [Output("report-status", "children"),
             Output("export-status-inline", "children", allow_duplicate=True)],
            [Input("generate-regression-report-btn", "n_clicks"),
             Input("generate-regression-report-btn-inline", "n_clicks")],
            prevent_initial_call=True
        )
        def generate_regression_report(n_clicks, n_clicks_inline):
            """Generate a comprehensive regression report."""
            # Check which button was clicked
            trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
            
            if (not n_clicks and not n_clicks_inline) or not self.regression_visualization:
                return "No data for report generation", dash.no_update
                
            # Get the metrics data and regressions from the data cache
            metrics_data = {}
            regressions_by_metric = {}
            
            for key, regressions in self.data_cache["regression_analysis"]["regressions_by_metric"].items():
                # Extract metric name from combined key (format: "metric_name_hardware")
                parts = key.split("_")
                if len(parts) > 1:
                    metric_name = parts[0]
                    
                    # Add to metrics data if available
                    for perf_metric, hw_data in self.data_cache["performance_metrics"].items():
                        if perf_metric == metric_name and key in hw_data:
                            metrics_data[key] = hw_data[key]
                            regressions_by_metric[key] = regressions
            
            if not metrics_data or not regressions_by_metric:
                return "No regression data available for report generation"
                
            try:
                # Generate the report
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(self.regression_output_dir, f"regression_report_{timestamp}.html")
                
                generated_path = self.regression_visualization.create_regression_summary_report(
                    metrics_data,
                    regressions_by_metric,
                    output_path=report_path,
                    include_plots=True
                )
                
                if generated_path:
                    return f"Report generated: {os.path.basename(generated_path)}"
                else:
                    return "Failed to generate report"
            except Exception as e:
                logger.error(f"Error generating regression report: {e}")
                return f"Error generating report: {str(e)}"
        
        # Callback for running comparative regression analysis
        @self.app.callback(
            Output("correlation-chart", "figure"),
            [Input("run-correlation-analysis-btn", "n_clicks"),
             Input("dashboard-data-store", "data")],
            prevent_initial_call=True
        )
        def run_correlation_analysis(n_clicks, data):
            """Run correlation analysis on all metrics."""
            if not data or not self.regression_detector:
                # Return empty figure if no data or regression detector
                return go.Figure().update_layout(
                    title="No correlation data available",
                    template="plotly_dark" if self.theme == "dark" else "plotly"
                )
            
            # Check if triggered by button click
            is_button_click = dash.callback_context.triggered_id == "run-correlation-analysis-btn"
            
            # Only run analysis if triggered by button click
            if not is_button_click:
                # Return existing analysis if available
                correlation_analysis = self.data_cache["regression_analysis"].get("correlation_analysis")
                if correlation_analysis and "visualization" in correlation_analysis:
                    return go.Figure(correlation_analysis["visualization"])
                    
                # Otherwise return empty figure
                return go.Figure().update_layout(
                    title="Click 'Run Correlation Analysis' button to analyze correlations",
                    template="plotly_dark" if self.theme == "dark" else "plotly"
                )
            
            # Get performance metrics data
            performance = data.get("performance_metrics", {})
            
            # Prepare time series data for all metrics
            # We'll use the first hardware type for each metric for simplicity
            metrics_data = {}
            
            for metric, hw_data in performance.items():
                if hw_data:
                    # Get the first hardware type
                    first_hw = list(hw_data.keys())[0]
                    series_data = hw_data[first_hw]
                    
                    if "timestamps" in series_data and "values" in series_data:
                        metrics_data[metric] = series_data
            
            # Run correlation analysis - use enhanced visualization if available
            if metrics_data:
                try:
                    if self.regression_visualization and self.enhanced_visualization:
                        # Load any cached regressions by metric
                        regressions_by_metric = self.data_cache["regression_analysis"].get("regressions_by_metric", {})
                        
                        # Prepare regressions by metric (format may need adjustment)
                        formatted_regressions = {}
                        for metric in metrics_data.keys():
                            for key, regs in regressions_by_metric.items():
                                if key.startswith(f"{metric}_"):
                                    formatted_regressions[metric] = regs
                                    break
                        
                        # Create comparative visualization with enhanced features
                        correlation_fig = self.regression_visualization.create_comparative_regression_visualization(
                            metrics_data,
                            formatted_regressions,
                            title="Comparative Regression Analysis"
                        )
                        
                        if correlation_fig:
                            # Store in data cache and return
                            self.data_cache["regression_analysis"]["correlation_analysis"] = {
                                "visualization": correlation_fig
                            }
                            return go.Figure(correlation_fig)
                    
                    # Fallback to basic correlation if enhanced visualization not available
                    elif self.regression_detector:
                        correlation_analysis = self.regression_detector.create_correlation_analysis(metrics_data)
                        
                        if correlation_analysis:
                            # Store in data cache
                            self.data_cache["regression_analysis"]["correlation_analysis"] = correlation_analysis
                            
                            # Return visualization
                            if "visualization" in correlation_analysis:
                                return go.Figure(correlation_analysis["visualization"])
                            
                except Exception as e:
                    logger.error(f"Error in correlation analysis: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Return empty figure if analysis failed
            return go.Figure().update_layout(
                title="Correlation analysis failed or insufficient data",
                template="plotly_dark" if self.theme == "dark" else "plotly"
            )
        
        # Callback for updating system health chart
        @self.app.callback(
            Output("system-health-chart", "figure"),
            Input("dashboard-data-store", "data"),
            prevent_initial_call=False
        )
        def update_system_health_chart(data):
            """Update system health chart."""
            if not data or "system_health" not in data:
                # Return empty figure if no data
                return go.Figure().update_layout(
                    title="No system health data available",
                    template="plotly_dark" if self.theme == "dark" else "plotly"
                )
            
            # Extract data
            system_health = data.get("system_health", {})
            
            # Create figure
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add CPU usage trace
            if "cpu_usage" in system_health:
                fig.add_trace(
                    go.Scatter(
                        x=system_health["timestamps"],
                        y=system_health["cpu_usage"],
                        name="CPU Usage",
                        line=dict(color="#3498db", width=2),
                    ),
                    secondary_y=False,
                )
            
            # Add memory usage trace
            if "memory_usage" in system_health:
                fig.add_trace(
                    go.Scatter(
                        x=system_health["timestamps"],
                        y=system_health["memory_usage"],
                        name="Memory Usage",
                        line=dict(color="#2ecc71", width=2),
                    ),
                    secondary_y=False,
                )
            
            # Add network usage trace
            if "network_usage" in system_health:
                fig.add_trace(
                    go.Scatter(
                        x=system_health["timestamps"],
                        y=system_health["network_usage"],
                        name="Network (MB/s)",
                        line=dict(color="#e74c3c", width=2),
                    ),
                    secondary_y=True,
                )
            
            # Update layout
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
                template="plotly_dark" if self.theme == "dark" else "plotly",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                xaxis=dict(showgrid=False),
            )
            
            # Update y-axes titles
            fig.update_yaxes(title_text="Percentage", secondary_y=False)
            fig.update_yaxes(title_text="Network MB/s", secondary_y=True)
            
            return fig
        
        # Callback for updating performance time chart
        @self.app.callback(
            Output("performance-time-chart", "figure"),
            Input("dashboard-data-store", "data"),
            prevent_initial_call=False
        )
        def update_performance_time_chart(data):
            """Update performance time chart."""
            if not data or "performance_metrics" not in data:
                # Return empty figure if no data
                return go.Figure().update_layout(
                    title="No performance data available",
                    template="plotly_dark" if self.theme == "dark" else "plotly"
                )
            
            # Extract data
            performance = data.get("performance_metrics", {})
            
            # Get selected metrics
            metrics = data.get("selected_metrics", ["throughput", "latency", "memory_usage"])
            
            # Create figure with subplots
            fig = make_subplots(
                rows=len(metrics),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[m.replace("_", " ").title() for m in metrics]
            )
            
            # Add traces for each metric
            for i, metric in enumerate(metrics):
                if metric in performance:
                    for series_name, series_data in performance[metric].items():
                        fig.add_trace(
                            go.Scatter(
                                x=series_data.get("timestamps", []),
                                y=series_data.get("values", []),
                                name=f"{series_name} - {metric}",
                                line=dict(width=2),
                            ),
                            row=i+1, col=1
                        )
            
            # Update layout
            fig.update_layout(
                height=300 * len(metrics),
                template="plotly_dark" if self.theme == "dark" else "plotly",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            return fig
        
        # Callback for updating the regression visualization options
        @self.app.callback(
            Output("regression-chart", "figure"),
            [Input("regression-viz-options", "value"),
             Input("dashboard-data-store", "data")],
            [State("regression-metric-dropdown", "value")],
            prevent_initial_call=True
        )
        def update_regression_visualization_options(viz_options, data, selected_metric):
            """Update the regression visualization with the selected options."""
            if not data or not self.regression_detector or selected_metric is None:
                return go.Figure().update_layout(
                    title="No regression data available",
                    template="plotly_dark" if self.theme == "dark" else "plotly"
                )
            
            # Get performance metrics data and regressions
            performance = data.get("performance_metrics", {})
            regressions_by_metric = self.data_cache["regression_analysis"].get("regressions_by_metric", {})
            
            # Find the metric data
            if selected_metric in performance:
                # For each hardware series (use the first one)
                for hw, series_data in performance[selected_metric].items():
                    if "timestamps" in series_data and "values" in series_data:
                        time_series_key = f"{selected_metric}_{hw}"
                        first_data = series_data
                        
                        # Get the regressions for this time series
                        regressions = regressions_by_metric.get(time_series_key, [])
                        
                        # Create visualization with selected options if enhanced visualization is available
                        if self.regression_visualization and self.enhanced_visualization:
                            try:
                                # Parse visualization options
                                include_ci = "ci" in viz_options
                                include_trend = "trend" in viz_options
                                include_annotations = "annotations" in viz_options
                                
                                # Create visualization
                                fig_dict = self.regression_visualization.create_interactive_regression_figure(
                                    first_data,
                                    regressions,
                                    selected_metric,
                                    title=f"Regression Analysis for {selected_metric}",
                                    include_confidence_intervals=include_ci,
                                    include_trend_lines=include_trend,
                                    include_annotations=include_annotations
                                )
                                
                                if fig_dict:
                                    return go.Figure(fig_dict)
                            except Exception as e:
                                logger.error(f"Error updating regression visualization: {e}")
            
            # Return the existing figure from the data cache if update failed
            report = self.data_cache["regression_analysis"].get("regression_report")
            return go.Figure().update_layout(
                title=f"Could not update visualization options for {selected_metric}",
                template="plotly_dark" if self.theme == "dark" else "plotly"
            )
        
        # Callback for exporting regression visualization
        @self.app.callback(
            [Output("regression-export-info", "children"),
             Output("regression-export-info", "style")],
            Input("export-regression-viz-btn", "n_clicks"),
            [State("regression-chart", "figure"),
             State("regression-metric-dropdown", "value")],
            prevent_initial_call=True
        )
        def export_regression_visualization(n_clicks, figure, selected_metric):
            """Export the current regression visualization."""
            if n_clicks is None or not figure or selected_metric is None:
                return None, {"display": "none"}
            
            if self.regression_visualization:
                try:
                    # Export as HTML by default
                    export_path = self.regression_visualization.export_regression_visualization(
                        figure,
                        format="html"
                    )
                    
                    if export_path:
                        return f"Visualization exported to: {export_path}", {"display": "block"}
                except Exception as e:
                    logger.error(f"Error exporting regression visualization: {e}")
                    return f"Error exporting visualization: {str(e)}", {"display": "block", "backgroundColor": "#f8d7da", "color": "#721c24"}
            
            return "Regression visualization export not available", {"display": "block", "backgroundColor": "#fff3cd", "color": "#856404"}
        
        # Callback for generating regression report
        @self.app.callback(
            [Output("regression-export-info", "children", allow_duplicate=True),
             Output("regression-export-info", "style", allow_duplicate=True)],
            Input("generate-regression-report-btn", "n_clicks"),
            Input("dashboard-data-store", "data"),
            prevent_initial_call=True
        )
        def generate_regression_report(n_clicks, data):
            """Generate a comprehensive regression report."""
            if n_clicks is None or not data:
                return None, {"display": "none"}
            
            if self.regression_visualization:
                try:
                    # Get metrics data and regressions
                    performance = data.get("performance_metrics", {})
                    regressions_by_metric = self.data_cache["regression_analysis"].get("regressions_by_metric", {})
                    
                    # Prepare metrics data for the report
                    metrics_data = {}
                    
                    for metric, hw_data in performance.items():
                        if hw_data:
                            # Get the first hardware type
                            first_hw = list(hw_data.keys())[0]
                            series_data = hw_data[first_hw]
                            
                            if "timestamps" in series_data and "values" in series_data:
                                metrics_data[metric] = series_data
                    
                    # Format regressions by metric
                    formatted_regressions = {}
                    for metric in metrics_data.keys():
                        for key, regs in regressions_by_metric.items():
                            if key.startswith(f"{metric}_"):
                                formatted_regressions[metric] = regs
                                break
                    
                    # Generate report
                    report_path = self.regression_visualization.create_regression_summary_report(
                        metrics_data,
                        formatted_regressions,
                        include_plots=True
                    )
                    
                    if report_path:
                        return f"Regression report generated: {report_path}", {"display": "block"}
                except Exception as e:
                    logger.error(f"Error generating regression report: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return f"Error generating report: {str(e)}", {"display": "block", "backgroundColor": "#f8d7da", "color": "#721c24"}
            
            return "Regression report generation not available", {"display": "block", "backgroundColor": "#fff3cd", "color": "#856404"}
            
        # Additional callbacks would be defined here for all other charts and UI elements
        # For brevity, I've only included a few key callbacks
    
    def _get_dashboard_data(self, metrics=None, models=None, hardware=None, time_range=None):
        """Get dashboard data based on filters.
        
        Args:
            metrics: List of metrics to include
            models: Models to include (or "all")
            hardware: Hardware to include (or "all")
            time_range: Time range to include
            
        Returns:
            Dictionary of dashboard data
        """
        # In a real implementation, this would fetch data from the database and result aggregator
        # For now, return mock data
        
        # Generate timestamps for the past hour
        now = datetime.datetime.now()
        if time_range == "1h":
            timestamps = [now - datetime.timedelta(minutes=i) for i in range(60, 0, -1)]
        elif time_range == "1d":
            timestamps = [now - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
        elif time_range == "7d":
            timestamps = [now - datetime.timedelta(days=i) for i in range(7, 0, -1)]
        else:  # 30d default
            timestamps = [now - datetime.timedelta(days=i) for i in range(30, 0, -1)]
            
        # Format timestamps as strings
        timestamp_strs = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
        
        # Generate random system health metrics
        system_health = {
            "timestamps": timestamp_strs,
            "cpu_usage": [random.uniform(10, 80) for _ in range(len(timestamps))],
            "memory_usage": [random.uniform(20, 90) for _ in range(len(timestamps))],
            "network_usage": [random.uniform(0.1, 5.0) for _ in range(len(timestamps))],
        }
        
        # Generate random performance metrics
        performance_metrics = {}
        
        if metrics is None:
            metrics = ["throughput", "latency", "memory_usage"]
            
        for metric in metrics:
            performance_metrics[metric] = {}
            
            # Generate data for different hardware types
            hw_types = ["cpu", "cuda", "webgpu", "webnn"] if hardware == "all" else [hardware]
            
            for hw in hw_types:
                if hw != "all":
                    # Generate random values based on metric
                    if metric == "throughput":
                        values = [random.uniform(50, 200) for _ in range(len(timestamps))]
                    elif metric == "latency":
                        values = [random.uniform(10, 100) for _ in range(len(timestamps))]
                    elif metric == "memory_usage":
                        values = [random.uniform(100, 1000) for _ in range(len(timestamps))]
                    elif metric == "cpu_usage":
                        values = [random.uniform(10, 80) for _ in range(len(timestamps))]
                    elif metric == "gpu_usage":
                        values = [random.uniform(20, 90) for _ in range(len(timestamps))]
                    else:
                        values = [random.uniform(0, 100) for _ in range(len(timestamps))]
                        
                    performance_metrics[metric][hw] = {
                        "timestamps": timestamp_strs,
                        "values": values
                    }
        
        # Return combined data
        return {
            "selected_metrics": metrics,
            "selected_models": models,
            "selected_hardware": hardware,
            "selected_time_range": time_range,
            "system_health": system_health,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    async def start(self):
        """Start the dashboard."""
        if DASH_AVAILABLE and self.app is not None:
            try:
                # Start WebSocket server if available
                if AIOHTTP_AVAILABLE:
                    await self._start_websocket_server()
                
                # Start Dash app
                self.app.run_server(
                    host=self.host,
                    port=self.port,
                    debug=self.debug
                )
                logger.info(f"Dashboard started at http://{self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Error starting dashboard: {e}")
        else:
            logger.error("Dash not available. Cannot start dashboard.")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates."""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available. Cannot start WebSocket server.")
            return
        
        # Import WebSocket server components
        from aiohttp import web, WSMsgType
        
        # WebSocket handler
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            
            # Register client
            self.websocket_clients.add(ws)
            logger.info(f"WebSocket client connected. Total clients: {len(self.websocket_clients)}")
            
            try:
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        if msg.data == "close":
                            await ws.close()
                        else:
                            # Handle client messages if needed
                            pass
                    elif msg.type == WSMsgType.ERROR:
                        logger.error(f"WebSocket connection closed with error: {ws.exception()}")
            finally:
                # Unregister client
                self.websocket_clients.remove(ws)
                logger.info(f"WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")
            
            return ws
        
        # Create app
        app = web.Application()
        app.router.add_get("/ws", websocket_handler)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Bind to different port than Dash
        site = web.TCPSite(runner, self.host, self.port + 1)
        await site.start()
        
        logger.info(f"WebSocket server started at ws://{self.host}:{self.port + 1}/ws")
        
        # Start broadcast task
        # TODO: Replace with task group - asyncio.create_task(self._broadcast_updates())
    
    async def _broadcast_updates(self):
        """Broadcast updates to WebSocket clients."""
        while True:
            try:
                # Only broadcast if there are clients
                if self.websocket_clients:
                    # Get latest data
                    data = self._get_dashboard_data()
                    
                    # Broadcast to all clients
                    for ws in self.websocket_clients:
                        await ws.send_json(data)
                    
                    logger.debug(f"Broadcast update to {len(self.websocket_clients)} clients")
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
            
            # Wait before next broadcast
            await anyio.sleep(self.config["refresh_interval"])
    
    def create_custom_dashboard(self, 
                               title: str, 
                               layout: List[Dict[str, Any]],
                               data_sources: Dict[str, str] = None,
                               output_path: Optional[str] = None):
        """Create a custom dashboard with the specified layout.
        
        Args:
            title: Dashboard title
            layout: Dashboard layout as a list of component definitions
            data_sources: Optional mapping of component IDs to data source IDs
            output_path: Optional path to save the dashboard
            
        Returns:
            Path to the generated dashboard
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available. Cannot create custom dashboard.")
            return None
        
        try:
            # Generate default output path if not provided
            if output_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"custom_dashboard_{timestamp}.html")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create dashboard HTML
            dashboard_html = self._generate_custom_dashboard_html(title, layout, data_sources)
            
            # Write to file
            with open(output_path, "w") as f:
                f.write(dashboard_html)
                
            logger.info(f"Custom dashboard created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating custom dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_custom_dashboard_html(self, title, layout, data_sources):
        """Generate HTML for a custom dashboard.
        
        Args:
            title: Dashboard title
            layout: Dashboard layout
            data_sources: Data source mappings
            
        Returns:
            Dashboard HTML string
        """
        # Get color scheme based on theme
        colors = self.config["color_schemes"][self.theme]
        
        # Start HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: {colors["background"]};
                    color: {colors["text"]};
                }}
                .dashboard-container {{
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }}
                .dashboard-header {{
                    padding: 10px;
                    background-color: {colors["primary"]};
                    color: white;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .dashboard-title {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 0;
                }}
                .dashboard-timestamp {{
                    font-size: 14px;
                    opacity: 0.8;
                    margin-top: 5px;
                }}
                .card {{
                    background-color: {colors["secondary"]};
                    color: {colors["text"]};
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                }}
                .card-header {{
                    background-color: rgba(0, 0, 0, 0.1);
                    padding: 10px 15px;
                    font-weight: bold;
                }}
                .card-body {{
                    padding: 15px;
                }}
                .chart-container {{
                    width: 100%;
                    height: 400px;
                }}
                .grid-container {{
                    display: grid;
                    grid-template-columns: repeat(12, 1fr);
                    grid-auto-rows: minmax(100px, auto);
                    grid-gap: 20px;
                }}
                .grid-item {{
                    grid-column: span 12;
                }}
                @media (min-width: 768px) {{
                    .grid-item.col-md-6 {{
                        grid-column: span 6;
                    }}
                    .grid-item.col-md-4 {{
                        grid-column: span 4;
                    }}
                    .grid-item.col-md-3 {{
                        grid-column: span 3;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1 class="dashboard-title">{title}</h1>
                    <div class="dashboard-timestamp">Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>
                
                <div class="grid-container">
        """
        
        # Add layout components
        for component in layout:
            component_type = component.get("type", "chart")
            component_title = component.get("title", "Chart")
            component_width = component.get("width", 12)  # Default full width
            component_height = component.get("height", 400)
            component_id = component.get("id", f"component_{hash(component_title)}")
            
            # Start component
            html += f"""
                    <div class="grid-item col-md-{component_width}">
                        <div class="card">
                            <div class="card-header">{component_title}</div>
                            <div class="card-body">
            """
            
            # Add component content based on type
            if component_type == "chart":
                html += f"""
                                <div class="chart-container" id="{component_id}" style="height: {component_height}px;"></div>
                """
            elif component_type == "metric":
                html += f"""
                                <div class="text-center">
                                    <h2 id="{component_id}_value" class="display-4">0</h2>
                                    <p id="{component_id}_label" class="text-muted">{component.get("label", "Metric")}</p>
                                </div>
                """
            elif component_type == "table":
                html += f"""
                                <div class="table-responsive">
                                    <table class="table table-striped" id="{component_id}">
                                        <thead>
                                            <tr>
                                                <th>Column 1</th>
                                                <th>Column 2</th>
                                                <th>Column 3</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr><td colspan="3">No data available</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                """
            elif component_type == "text":
                html += f"""
                                <div id="{component_id}" class="content-container">
                                    {component.get("content", "No content available")}
                                </div>
                """
            
            # End component
            html += """
                            </div>
                        </div>
                    </div>
            """
        
        # Add JavaScript for data loading and chart creation
        html += """
                </div>
            </div>
            
            <script>
                // Dashboard data
                const dashboardData = {};
                
                // Function to create charts
                function createCharts() {
                    // Create charts based on layout configuration
                    // In a real implementation, this would use the layout and data_sources
                    // to create appropriate charts
                    
                    // Example chart
                    const exampleData = {
                        x: Array.from({length: 10}, (_, i) => i),
                        y: Array.from({length: 10}, () => Math.random() * 100),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Example Data'
                    };
                    
                    // Create charts for all containers
                    document.querySelectorAll('.chart-container').forEach((container) => {
                        Plotly.newPlot(container.id, [exampleData], {
                            margin: { t: 10, r: 10, b: 40, l: 40 },
                            template: 'plotly_dark',
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: '#f8f9fa' },
                            xaxis: { gridcolor: '#444444' },
                            yaxis: { gridcolor: '#444444' }
                        });
                    });
                }
                
                // Initialize dashboard
                createCharts();
                
                // In a real implementation, this would connect to a WebSocket for real-time updates
                // and update the charts with new data
            </script>
        </body>
        </html>
        """
        
        return html
    
    def export_dashboard(self, output_format="html", output_path=None):
        """Export the current dashboard.
        
        Args:
            output_format: Format to export to (html, pdf, png)
            output_path: Path to save the export
            
        Returns:
            Path to the exported dashboard
        """
        # In a real implementation, this would export the dashboard to the specified format
        # For now, just return the output path
        
        # Generate default output path if not provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"dashboard_export_{timestamp}.{output_format}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # For demonstration purposes, generate a simple export
        if output_format == "html":
            # Create a simple HTML export
            with open(output_path, "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Dashboard Export</title>
                </head>
                <body>
                    <h1>Dashboard Export</h1>
                    <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>This is a placeholder for the dashboard export.</p>
                </body>
                </html>
                """)
        else:
            # Other formats would require additional libraries
            logger.warning(f"Export to {output_format} not implemented yet.")
            return None
        
        logger.info(f"Dashboard exported to: {output_path}")
        return output_path
    
    def stop(self):
        """Stop the dashboard."""
        # In a real implementation, this would stop the dashboard server
        logger.info("Dashboard stopped")
    
    @staticmethod
    def run_from_command_line():
        """Run the dashboard from the command line."""
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Run the enhanced visualization dashboard")
        parser.add_argument("--host", default="localhost", help="Host to bind to")
        parser.add_argument("--port", type=int, default=8082, help="Port to bind to")
        parser.add_argument("--db-path", default="benchmark_db.duckdb", help="Path to DuckDB database")
        parser.add_argument("--output-dir", default="./visualizations/dashboard", help="Output directory for visualizations")
        parser.add_argument("--theme", default="dark", choices=["light", "dark"], help="Dashboard theme")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--browser", action="store_true", help="Open dashboard in browser")
        args = parser.parse_args()
        
        # Create and start dashboard
        dashboard = EnhancedVisualizationDashboard(
            db_path=args.db_path,
            output_dir=args.output_dir,
            host=args.host,
            port=args.port,
            debug=args.debug,
            theme=args.theme
        )
        
        # Start the dashboard
        import asyncio
        loop = # TODO: Remove event loop management - asyncio.get_event_loop()
        
        try:
            # Open browser if requested
            if args.browser:
                import webbrowser
                webbrowser.open(f"http://{args.host}:{args.port}")
            
            # Start dashboard
            loop.run_until_complete(dashboard.start())
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        finally:
            dashboard.stop()

# For command-line usage
if __name__ == "__main__":
    import random  # For mock data generation
    EnhancedVisualizationDashboard.run_from_command_line()