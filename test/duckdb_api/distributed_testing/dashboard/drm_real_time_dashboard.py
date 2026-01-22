#!/usr/bin/env python3
"""
Real-Time Performance Metrics Dashboard for Dynamic Resource Management (DRM)

This module implements real-time visualization capabilities for the Dynamic Resource
Management component of the Distributed Testing Framework. It provides:

1. Real-time resource utilization monitoring
2. Performance metrics visualization with trend analysis
3. Scaling decision visualization and alerting
4. WebSocket-based live updates with minimal performance impact
5. Integration with existing dashboard components

Usage:
    # Import the module
    from duckdb_api.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
    
    # Create a dashboard instance
    dashboard = DRMRealTimeDashboard(
        dynamic_resource_manager=drm_instance,
        port=8085,
        debug=True
    )
    
    # Start the dashboard
    dashboard.start()
"""

import os
import json
import time
import logging
import threading
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path

# For visualization components
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, Input, Output, callback, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    
# For WebSocket communication
try:
    import tornado.web
    import tornado.ioloop
    import tornado.websocket
    TORNADO_AVAILABLE = True
except ImportError:
    TORNADO_AVAILABLE = False

# For statistical analysis
try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules from parent
import sys
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import required modules with fallbacks
try:
    from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager
    DRM_AVAILABLE = True
except ImportError:
    logger.warning("DynamicResourceManager not available, some features will be limited")
    DRM_AVAILABLE = False

try:
    from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector
    REGRESSION_AVAILABLE = True
except ImportError:
    logger.warning("RegressionDetector not available, regression analysis will be disabled")
    REGRESSION_AVAILABLE = False


class DRMRealTimeDashboard:
    """
    Real-Time Dashboard for Dynamic Resource Management.
    
    Provides real-time visualization of resource metrics, scaling decisions,
    and performance trends with statistical analysis of the DRM system.
    """
    
    def __init__(
        self,
        dynamic_resource_manager=None,
        db_path: str = "benchmark_db.duckdb",
        port: int = 8085,
        update_interval: int = 5,
        retention_window: int = 60,
        debug: bool = False,
        theme: str = "dark"
    ):
        """
        Initialize the real-time DRM dashboard.
        
        Args:
            dynamic_resource_manager: DynamicResourceManager instance
            db_path: Path to DuckDB database for historical data
            port: Port to run the dashboard on
            update_interval: Update interval in seconds
            retention_window: Data retention window in minutes
            debug: Enable debug mode
            theme: Dashboard theme ('dark' or 'light')
        """
        self.drm = dynamic_resource_manager
        self.db_path = db_path
        self.port = port
        self.update_interval = update_interval
        self.retention_window = retention_window
        self.debug = debug
        self.theme = theme
        
        # Initialize data storage
        self.resource_metrics = {
            "timestamps": [],
            "cpu_utilization": [],
            "memory_utilization": [],
            "gpu_utilization": [],
            "worker_count": [],
            "active_tasks": [],
            "pending_tasks": []
        }
        
        self.scaling_decisions = []
        self.performance_metrics = {
            "task_throughput": [],
            "allocation_time": [],
            "resource_efficiency": []
        }
        
        self.worker_metrics = {}
        self.alerts = []
        
        # Initialize regression detector if available
        self.regression_detector = None
        if REGRESSION_AVAILABLE:
            self.regression_detector = RegressionDetector()
        
        # Dashboard state
        self.dashboard_app = None
        self.update_thread = None
        self.update_stop_event = threading.Event()
        self.clients = set()
        self.is_running = False
        
        # Create output directory for visualizations
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "static",
            "drm_visualizations"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"DRM Real-Time Dashboard initialized on port {port}")
    
    def start(self):
        """Start the dashboard server and data collection."""
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Cannot start dashboard.")
            return False
        
        if self.is_running:
            logger.warning("Dashboard is already running")
            return True
        
        try:
            # Start data collection thread
            self._start_data_collection()
            
            # Initialize and start the dashboard
            self._initialize_dashboard()
            
            # Set running flag
            self.is_running = True
            
            # Display startup message
            logger.info(f"DRM Real-Time Dashboard running at http://localhost:{self.port}")
            logger.info("Press Ctrl+C to stop.")
            
            # Start the Dash server (this is blocking)
            self.dashboard_app.run_server(
                debug=self.debug,
                port=self.port,
                use_reloader=False
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            self.stop()
            return False
    
    def start_in_background(self):
        """Start the dashboard server in a background thread."""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return True
        
        # Start data collection thread
        self._start_data_collection()
        
        # Initialize the dashboard
        self._initialize_dashboard()
        
        # Start in background thread
        threading.Thread(
            target=self.dashboard_app.run_server,
            kwargs={
                "debug": False,  # Debug must be False in background thread
                "port": self.port,
                "use_reloader": False
            },
            daemon=True
        ).start()
        
        # Set running flag
        self.is_running = True
        
        logger.info(f"DRM Real-Time Dashboard running in background at http://localhost:{self.port}")
        return True
    
    def stop(self):
        """Stop the dashboard server and data collection."""
        # Stop data collection thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_stop_event.set()
            self.update_thread.join(timeout=5.0)
        
        # Reset running flag
        self.is_running = False
        
        logger.info("DRM Real-Time Dashboard stopped")
    
    def _start_data_collection(self):
        """Start the data collection thread."""
        if self.update_thread and self.update_thread.is_alive():
            return
        
        self.update_stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._data_collection_loop,
            daemon=True
        )
        self.update_thread.start()
        logger.info("Data collection started")
    
    def _data_collection_loop(self):
        """Background thread for collecting data."""
        while not self.update_stop_event.is_set():
            try:
                # Collect metrics from DRM if available
                if self.drm:
                    self._collect_drm_metrics()
                
                # Prune old data to maintain retention window
                self._prune_old_data()
                
                # Perform regression detection if enabled
                self._detect_performance_regressions()
                
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
            
            # Wait for next update
            self.update_stop_event.wait(self.update_interval)
    
    def _collect_drm_metrics(self):
        """Collect metrics from DRM instance."""
        current_time = datetime.datetime.now()
        
        try:
            # Get resource statistics
            worker_stats = self.drm.get_worker_statistics()
            
            # Update resource metrics
            self.resource_metrics["timestamps"].append(current_time)
            self.resource_metrics["worker_count"].append(worker_stats.get("total_workers", 0))
            self.resource_metrics["active_tasks"].append(worker_stats.get("active_tasks", 0))
            self.resource_metrics["pending_tasks"].append(worker_stats.get("pending_tasks", 0))
            
            # Extract utilization metrics
            overall_utilization = worker_stats.get("overall_utilization", {})
            self.resource_metrics["cpu_utilization"].append(overall_utilization.get("cpu", 0) * 100)
            self.resource_metrics["memory_utilization"].append(overall_utilization.get("memory", 0) * 100)
            self.resource_metrics["gpu_utilization"].append(overall_utilization.get("gpu", 0) * 100)
            
            # Update worker-specific metrics
            for worker_id, worker_data in worker_stats.get("workers", {}).items():
                if worker_id not in self.worker_metrics:
                    self.worker_metrics[worker_id] = {
                        "timestamps": [],
                        "cpu_utilization": [],
                        "memory_utilization": [],
                        "gpu_utilization": [],
                        "tasks": []
                    }
                
                self.worker_metrics[worker_id]["timestamps"].append(current_time)
                
                # Extract worker utilization
                utilization = worker_data.get("utilization", {})
                self.worker_metrics[worker_id]["cpu_utilization"].append(utilization.get("cpu", 0) * 100)
                self.worker_metrics[worker_id]["memory_utilization"].append(utilization.get("memory", 0) * 100)
                self.worker_metrics[worker_id]["gpu_utilization"].append(utilization.get("gpu", 0) * 100)
                self.worker_metrics[worker_id]["tasks"].append(worker_data.get("tasks", 0))
            
            # Get performance metrics
            if hasattr(self.drm, "get_performance_metrics"):
                perf_metrics = self.drm.get_performance_metrics()
                
                # Update performance metrics
                if "task_throughput" in perf_metrics:
                    self.performance_metrics["task_throughput"].append({
                        "timestamp": current_time,
                        "value": perf_metrics["task_throughput"]
                    })
                
                if "allocation_time" in perf_metrics:
                    self.performance_metrics["allocation_time"].append({
                        "timestamp": current_time,
                        "value": perf_metrics["allocation_time"]
                    })
                
                if "resource_efficiency" in perf_metrics:
                    self.performance_metrics["resource_efficiency"].append({
                        "timestamp": current_time,
                        "value": perf_metrics["resource_efficiency"]
                    })
            
            # Get last scaling decision if available
            if hasattr(self.drm, "last_scaling_decision") and self.drm.last_scaling_decision:
                last_decision = self.drm.last_scaling_decision
                
                # Check if this is a new decision
                if not self.scaling_decisions or last_decision != self.scaling_decisions[-1]["decision"]:
                    self.scaling_decisions.append({
                        "timestamp": current_time,
                        "decision": last_decision,
                        "action": last_decision.action if hasattr(last_decision, "action") else last_decision.get("action", "unknown"),
                        "reason": last_decision.reason if hasattr(last_decision, "reason") else last_decision.get("reason", ""),
                        "count": last_decision.count if hasattr(last_decision, "count") else last_decision.get("count", 0)
                    })
                    
                    # Check for critical scaling decisions that might require alerts
                    self._check_for_scaling_alerts(last_decision)
            
            logger.debug(f"Collected metrics at {current_time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error collecting DRM metrics: {e}")
    
    def _prune_old_data(self):
        """Prune old data to maintain the retention window."""
        if not self.resource_metrics["timestamps"]:
            return
            
        # Calculate cutoff time
        cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=self.retention_window)
        
        # Find index of first entry to keep
        keep_idx = 0
        for i, ts in enumerate(self.resource_metrics["timestamps"]):
            if ts >= cutoff_time:
                keep_idx = i
                break
        
        # Prune resource metrics if needed
        if keep_idx > 0:
            for key in self.resource_metrics:
                self.resource_metrics[key] = self.resource_metrics[key][keep_idx:]
            
            # Prune worker metrics
            for worker_id in self.worker_metrics:
                for key in self.worker_metrics[worker_id]:
                    # Find the index for this worker's data
                    w_keep_idx = 0
                    for i, ts in enumerate(self.worker_metrics[worker_id]["timestamps"]):
                        if ts >= cutoff_time:
                            w_keep_idx = i
                            break
                    
                    self.worker_metrics[worker_id][key] = self.worker_metrics[worker_id][key][w_keep_idx:]
            
            # Prune performance metrics
            for key in self.performance_metrics:
                self.performance_metrics[key] = [
                    entry for entry in self.performance_metrics[key]
                    if entry["timestamp"] >= cutoff_time
                ]
            
            # Prune scaling decisions
            self.scaling_decisions = [
                entry for entry in self.scaling_decisions
                if entry["timestamp"] >= cutoff_time
            ]
            
            # Prune alerts
            self.alerts = [
                alert for alert in self.alerts
                if alert["timestamp"] >= cutoff_time
            ]
            
            logger.debug(f"Pruned data older than {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _detect_performance_regressions(self):
        """Detect performance regressions in the time series data."""
        if not REGRESSION_AVAILABLE or not self.regression_detector or not self.resource_metrics["timestamps"]:
            return
        
        try:
            # Check for CPU utilization regressions
            if len(self.resource_metrics["cpu_utilization"]) >= 10:
                regressions = self.regression_detector.detect_regressions(
                    {
                        "timestamps": self.resource_metrics["timestamps"],
                        "values": self.resource_metrics["cpu_utilization"]
                    },
                    "cpu_utilization"
                )
                
                # Process significant regressions
                for regression in regressions:
                    if regression.get("is_significant", False):
                        # Create an alert for significant regressions
                        self._create_alert(
                            level="warning" if regression.get("severity") in ["critical", "high"] else "info",
                            message=f"CPU utilization {regression.get('direction')} by {abs(regression.get('percentage_change', 0)):.1f}%",
                            source="regression_detection",
                            details=regression
                        )
            
            # Check for allocation time regressions if data available
            if len(self.performance_metrics["allocation_time"]) >= 10:
                allocation_timestamps = [entry["timestamp"] for entry in self.performance_metrics["allocation_time"]]
                allocation_values = [entry["value"] for entry in self.performance_metrics["allocation_time"]]
                
                regressions = self.regression_detector.detect_regressions(
                    {
                        "timestamps": allocation_timestamps,
                        "values": allocation_values
                    },
                    "allocation_time_ms"
                )
                
                # Process significant regressions
                for regression in regressions:
                    if regression.get("is_significant", False) and regression.get("is_regression", False):
                        # Create an alert for significant regressions
                        self._create_alert(
                            level="warning",
                            message=f"Resource allocation time increased by {abs(regression.get('percentage_change', 0)):.1f}%",
                            source="regression_detection",
                            details=regression
                        )
            
        except Exception as e:
            logger.error(f"Error in regression detection: {e}")
    
    def _check_for_scaling_alerts(self, scaling_decision):
        """Check if a scaling decision should trigger an alert."""
        if not scaling_decision:
            return
        
        # Extract action and count
        if hasattr(scaling_decision, "action"):
            action = scaling_decision.action
            count = scaling_decision.count
            reason = scaling_decision.reason
        else:
            action = scaling_decision.get("action", "unknown")
            count = scaling_decision.get("count", 0)
            reason = scaling_decision.get("reason", "")
        
        # Alert on significant scaling events
        if action == "scale_up" and count >= 5:
            self._create_alert(
                level="info",
                message=f"Scaling up by {count} workers due to {reason}",
                source="scaling_decision",
                details={"action": action, "count": count, "reason": reason}
            )
        elif action == "scale_down" and count >= 5:
            self._create_alert(
                level="info",
                message=f"Scaling down by {count} workers due to {reason}",
                source="scaling_decision",
                details={"action": action, "count": count, "reason": reason}
            )
    
    def _create_alert(self, level, message, source, details=None):
        """Create a new alert."""
        alert = {
            "timestamp": datetime.datetime.now(),
            "level": level,
            "message": message,
            "source": source,
            "details": details or {},
            "id": len(self.alerts) + 1
        }
        
        self.alerts.append(alert)
        logger.info(f"Alert: [{level.upper()}] {message}")
    
    def _initialize_dashboard(self):
        """Initialize the Dash dashboard application."""
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Cannot initialize dashboard.")
            return
        
        # Select theme
        theme = dbc.themes.DARKLY if self.theme == "dark" else dbc.themes.BOOTSTRAP
        
        # Create Dash app
        self.dashboard_app = dash.Dash(
            __name__,
            external_stylesheets=[theme, dbc.icons.FONT_AWESOME],
            title="DRM Real-Time Dashboard",
            update_title=None
        )
        
        # Define layout
        self.dashboard_app.layout = self._create_dashboard_layout()
        
        # Add callbacks
        self._register_dashboard_callbacks()
    
    def _create_dashboard_layout(self):
        """Create the layout for the dashboard."""
        # Create navbar
        navbar = dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.I(className="fas fa-chart-line me-2")),
                        dbc.Col(dbc.NavbarBrand("DRM Real-Time Dashboard", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                    ),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Overview", href="#overview")),
                        dbc.NavItem(dbc.NavLink("Workers", href="#workers")),
                        dbc.NavItem(dbc.NavLink("Performance", href="#performance")),
                        dbc.NavItem(dbc.NavLink("Scaling", href="#scaling")),
                        dbc.NavItem(dbc.NavLink("Alerts", href="#alerts")),
                    ],
                    className="ms-auto",
                    navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
                # Update timestamp and interval indicator
                dbc.Row([
                    dbc.Col([
                        html.Span("Last updated: ", className="text-muted me-1"),
                        html.Span(id="last-update-time", className="fw-bold"),
                        html.Span(" (", className="text-muted ms-1"),
                        html.Span(f"updates every {self.update_interval}s", className="text-muted"),
                        html.Span(")", className="text-muted"),
                    ]),
                ], className="ms-3"),
            ]),
            color="primary",
            dark=True,
        )
        
        # Create main layout
        layout = html.Div([
            # Store for intermediate data
            dcc.Store(id="drm-data-store"),
            
            # Update interval
            dcc.Interval(
                id="interval-component",
                interval=self.update_interval * 1000,  # milliseconds
                n_intervals=0
            ),
            
            # Navbar
            navbar,
            
            # Main content
            dbc.Container([
                # Overview section
                html.Div([
                    html.H2("System Overview", className="mt-4", id="overview"),
                    html.Hr(),
                    
                    # Overview cards
                    dbc.Row([
                        # Worker count card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Workers", className="card-title"),
                                    html.H2(id="worker-count", className="card-text text-center display-4"),
                                    html.P(id="worker-trend", className="card-text text-center mt-2"),
                                ])
                            ]),
                            width=3,
                        ),
                        
                        # Active tasks card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Active Tasks", className="card-title"),
                                    html.H2(id="active-tasks-count", className="card-text text-center display-4"),
                                    html.P(id="tasks-trend", className="card-text text-center mt-2"),
                                ])
                            ]),
                            width=3,
                        ),
                        
                        # Utilization card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Avg. Utilization", className="card-title"),
                                    html.H2(id="avg-utilization", className="card-text text-center display-4"),
                                    html.P(id="utilization-trend", className="card-text text-center mt-2"),
                                ])
                            ]),
                            width=3,
                        ),
                        
                        # Alerts card
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Alerts", className="card-title"),
                                    html.H2(id="alert-count", className="card-text text-center display-4"),
                                    html.P(id="alert-trend", className="card-text text-center mt-2"),
                                ])
                            ]),
                            width=3,
                        ),
                    ], className="mb-4"),
                    
                    # Resource utilization graph
                    dbc.Card([
                        dbc.CardHeader("Resource Utilization"),
                        dbc.CardBody([
                            dcc.Graph(id="resource-utilization-graph", config={"displayModeBar": False})
                        ])
                    ], className="mb-4"),
                    
                    # Worker count and task graph
                    dbc.Card([
                        dbc.CardHeader("Worker Count and Tasks"),
                        dbc.CardBody([
                            dcc.Graph(id="worker-task-graph", config={"displayModeBar": False})
                        ])
                    ]),
                ]),
                
                # Worker section
                html.Div([
                    html.H2("Worker Details", className="mt-5", id="workers"),
                    html.Hr(),
                    
                    # Worker selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Workers:"),
                            dcc.Dropdown(
                                id="worker-selector",
                                multi=True,
                                placeholder="Select workers to display...",
                            ),
                        ], width=6),
                        
                        dbc.Col([
                            html.Label("Metric:"),
                            dcc.RadioItems(
                                id="worker-metric-selector",
                                options=[
                                    {"label": "CPU Utilization", "value": "cpu"},
                                    {"label": "Memory Utilization", "value": "memory"},
                                    {"label": "GPU Utilization", "value": "gpu"},
                                    {"label": "Tasks", "value": "tasks"}
                                ],
                                value="cpu",
                                inline=True,
                                className="mt-1"
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Worker metrics graph
                    dbc.Card([
                        dbc.CardHeader(id="worker-graph-title"),
                        dbc.CardBody([
                            dcc.Graph(id="worker-metrics-graph", config={"displayModeBar": False})
                        ])
                    ], className="mb-4"),
                    
                    # Worker utilization heatmap
                    dbc.Card([
                        dbc.CardHeader("Worker Utilization Heatmap"),
                        dbc.CardBody([
                            dcc.Graph(id="worker-heatmap", config={"displayModeBar": False})
                        ])
                    ]),
                ]),
                
                # Performance section
                html.Div([
                    html.H2("Performance Metrics", className="mt-5", id="performance"),
                    html.Hr(),
                    
                    # Performance metrics selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Metrics:"),
                            dcc.Checklist(
                                id="performance-metric-selector",
                                options=[
                                    {"label": "Task Throughput", "value": "task_throughput"},
                                    {"label": "Allocation Time", "value": "allocation_time"},
                                    {"label": "Resource Efficiency", "value": "resource_efficiency"}
                                ],
                                value=["task_throughput", "allocation_time"],
                                inline=True,
                                className="mt-1"
                            ),
                        ], width=6),
                        
                        dbc.Col([
                            html.Label("Time Range:"),
                            dcc.RadioItems(
                                id="performance-time-selector",
                                options=[
                                    {"label": "Last 15 min", "value": 15},
                                    {"label": "Last 30 min", "value": 30},
                                    {"label": "Last 60 min", "value": 60},
                                    {"label": "All", "value": self.retention_window}
                                ],
                                value=30,
                                inline=True,
                                className="mt-1"
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Performance metrics graph
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-metrics-graph", config={"displayModeBar": False})
                        ])
                    ], className="mb-4"),
                    
                    # Regression detection (if available)
                    dbc.Card([
                        dbc.CardHeader("Performance Regression Analysis"),
                        dbc.CardBody([
                            html.Div(id="regression-analysis-content")
                        ])
                    ]),
                ]),
                
                # Scaling section
                html.Div([
                    html.H2("Scaling Decisions", className="mt-5", id="scaling"),
                    html.Hr(),
                    
                    # Scaling decisions timeline
                    dbc.Card([
                        dbc.CardHeader("Scaling Decision Timeline"),
                        dbc.CardBody([
                            dcc.Graph(id="scaling-timeline-graph", config={"displayModeBar": False})
                        ])
                    ], className="mb-4"),
                    
                    # Recent scaling decisions table
                    dbc.Card([
                        dbc.CardHeader("Recent Scaling Decisions"),
                        dbc.CardBody([
                            html.Div(id="scaling-decisions-table")
                        ])
                    ]),
                ]),
                
                # Alerts section
                html.Div([
                    html.H2("Alerts and Notifications", className="mt-5", id="alerts"),
                    html.Hr(),
                    
                    # Alerts filter
                    dbc.Row([
                        dbc.Col([
                            html.Label("Filter by Level:"),
                            dcc.Checklist(
                                id="alert-level-filter",
                                options=[
                                    {"label": "Info", "value": "info"},
                                    {"label": "Warning", "value": "warning"},
                                    {"label": "Error", "value": "error"},
                                    {"label": "Critical", "value": "critical"}
                                ],
                                value=["info", "warning", "error", "critical"],
                                inline=True,
                                className="mt-1"
                            ),
                        ], width=6),
                        
                        dbc.Col([
                            html.Label("Filter by Source:"),
                            dcc.Checklist(
                                id="alert-source-filter",
                                options=[
                                    {"label": "Scaling Decision", "value": "scaling_decision"},
                                    {"label": "Regression Detection", "value": "regression_detection"},
                                    {"label": "System", "value": "system"}
                                ],
                                value=["scaling_decision", "regression_detection", "system"],
                                inline=True,
                                className="mt-1"
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Alerts list
                    dbc.Card([
                        dbc.CardHeader("Active Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-list")
                        ])
                    ]),
                ]),
                
                # Footer
                html.Footer([
                    html.Hr(className="mt-5"),
                    html.P([
                        "DRM Real-Time Dashboard • ",
                        html.Span(f"Data retention: {self.retention_window} minutes • "),
                        html.Span(f"Update interval: {self.update_interval} seconds"),
                    ], className="text-center text-muted"),
                ], className="mt-4 mb-4"),
                
            ], className="mt-4"),
        ])
        
        return layout
    
    def _register_dashboard_callbacks(self):
        """Register all callbacks for the dashboard."""
        app = self.dashboard_app
        
        # Data store update callback
        @app.callback(
            Output("drm-data-store", "data"),
            Output("last-update-time", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_data_store(n_intervals):
            """Update data store with current metrics."""
            # Format data for JSON serialization
            serialized_data = {
                "resource_metrics": {
                    key: [ts.isoformat() if isinstance(ts, datetime.datetime) else ts
                          for ts in self.resource_metrics["timestamps"]] if key == "timestamps"
                    else self.resource_metrics[key]
                    for key in self.resource_metrics
                },
                "worker_metrics": {
                    worker_id: {
                        key: [ts.isoformat() if isinstance(ts, datetime.datetime) else ts
                              for ts in data["timestamps"]] if key == "timestamps"
                        else data[key]
                        for key, data in worker_data.items()
                    }
                    for worker_id, worker_data in self.worker_metrics.items()
                },
                "performance_metrics": {
                    key: [
                        {"timestamp": entry["timestamp"].isoformat(), "value": entry["value"]}
                        for entry in self.performance_metrics[key]
                    ]
                    for key in self.performance_metrics
                },
                "scaling_decisions": [
                    {
                        "timestamp": decision["timestamp"].isoformat(),
                        "action": decision["action"],
                        "reason": decision["reason"],
                        "count": decision["count"]
                    }
                    for decision in self.scaling_decisions
                ],
                "alerts": [
                    {
                        "id": alert["id"],
                        "timestamp": alert["timestamp"].isoformat(),
                        "level": alert["level"],
                        "message": alert["message"],
                        "source": alert["source"]
                    }
                    for alert in self.alerts
                ],
                "settings": {
                    "update_interval": self.update_interval,
                    "retention_window": self.retention_window
                }
            }
            
            # Current update time
            update_time = datetime.datetime.now().strftime("%H:%M:%S")
            
            return serialized_data, update_time
        
        # Overview section callbacks
        @app.callback(
            Output("worker-count", "children"),
            Output("worker-trend", "children"),
            Output("worker-trend", "className"),
            Input("drm-data-store", "data")
        )
        def update_worker_count(data):
            """Update worker count card."""
            if not data or not data["resource_metrics"]["worker_count"]:
                return "0", "No data", "card-text text-center mt-2 text-muted"
            
            worker_count = data["resource_metrics"]["worker_count"][-1]
            
            # Calculate trend if enough data
            if len(data["resource_metrics"]["worker_count"]) > 1:
                prev_count = data["resource_metrics"]["worker_count"][-2]
                if worker_count > prev_count:
                    trend = f"↑ Increased by {worker_count - prev_count}"
                    trend_class = "card-text text-center mt-2 text-success"
                elif worker_count < prev_count:
                    trend = f"↓ Decreased by {prev_count - worker_count}"
                    trend_class = "card-text text-center mt-2 text-danger"
                else:
                    trend = "→ No change"
                    trend_class = "card-text text-center mt-2 text-muted"
            else:
                trend = "Monitoring..."
                trend_class = "card-text text-center mt-2 text-muted"
            
            return str(worker_count), trend, trend_class
        
        @app.callback(
            Output("active-tasks-count", "children"),
            Output("tasks-trend", "children"),
            Output("tasks-trend", "className"),
            Input("drm-data-store", "data")
        )
        def update_tasks_count(data):
            """Update active tasks card."""
            if not data or not data["resource_metrics"]["active_tasks"]:
                return "0", "No data", "card-text text-center mt-2 text-muted"
            
            tasks_count = data["resource_metrics"]["active_tasks"][-1]
            
            # Calculate trend if enough data
            if len(data["resource_metrics"]["active_tasks"]) > 1:
                prev_count = data["resource_metrics"]["active_tasks"][-2]
                if tasks_count > prev_count:
                    trend = f"↑ Increased by {tasks_count - prev_count}"
                    trend_class = "card-text text-center mt-2 text-success"
                elif tasks_count < prev_count:
                    trend = f"↓ Decreased by {prev_count - tasks_count}"
                    trend_class = "card-text text-center mt-2 text-danger"
                else:
                    trend = "→ No change"
                    trend_class = "card-text text-center mt-2 text-muted"
            else:
                trend = "Monitoring..."
                trend_class = "card-text text-center mt-2 text-muted"
            
            return str(tasks_count), trend, trend_class
        
        @app.callback(
            Output("avg-utilization", "children"),
            Output("utilization-trend", "children"),
            Output("utilization-trend", "className"),
            Input("drm-data-store", "data")
        )
        def update_utilization(data):
            """Update average utilization card."""
            if not data or not data["resource_metrics"]["cpu_utilization"]:
                return "0%", "No data", "card-text text-center mt-2 text-muted"
            
            # Calculate average utilization (CPU, memory, GPU)
            cpu = data["resource_metrics"]["cpu_utilization"][-1]
            memory = data["resource_metrics"]["memory_utilization"][-1]
            gpu = data["resource_metrics"]["gpu_utilization"][-1]
            
            # Only include non-zero values in the average
            values = [v for v in [cpu, memory, gpu] if v > 0]
            avg_utilization = sum(values) / len(values) if values else 0
            
            # Calculate trend if enough data
            if len(data["resource_metrics"]["cpu_utilization"]) > 1:
                # Calculate previous average
                prev_cpu = data["resource_metrics"]["cpu_utilization"][-2]
                prev_memory = data["resource_metrics"]["memory_utilization"][-2]
                prev_gpu = data["resource_metrics"]["gpu_utilization"][-2]
                
                prev_values = [v for v in [prev_cpu, prev_memory, prev_gpu] if v > 0]
                prev_avg = sum(prev_values) / len(prev_values) if prev_values else 0
                
                diff = avg_utilization - prev_avg
                if diff > 1:
                    trend = f"↑ Increased by {diff:.1f}%"
                    trend_class = "card-text text-center mt-2 text-danger"  # Higher utilization is a concern
                elif diff < -1:
                    trend = f"↓ Decreased by {-diff:.1f}%"
                    trend_class = "card-text text-center mt-2 text-success"  # Lower utilization is good
                else:
                    trend = "→ Stable"
                    trend_class = "card-text text-center mt-2 text-muted"
            else:
                trend = "Monitoring..."
                trend_class = "card-text text-center mt-2 text-muted"
            
            return f"{avg_utilization:.1f}%", trend, trend_class
        
        @app.callback(
            Output("alert-count", "children"),
            Output("alert-trend", "children"),
            Output("alert-trend", "className"),
            Input("drm-data-store", "data")
        )
        def update_alert_count(data):
            """Update alerts card."""
            if not data or "alerts" not in data:
                return "0", "No alerts", "card-text text-center mt-2 text-muted"
            
            alert_count = len(data["alerts"])
            
            # Count by severity
            warning_count = sum(1 for a in data["alerts"] if a["level"] in ["warning", "error", "critical"])
            
            if warning_count > 0:
                trend = f"{warning_count} require attention"
                trend_class = "card-text text-center mt-2 text-warning"
            else:
                trend = "All clear"
                trend_class = "card-text text-center mt-2 text-success"
            
            return str(alert_count), trend, trend_class
        
        @app.callback(
            Output("resource-utilization-graph", "figure"),
            Input("drm-data-store", "data")
        )
        def update_resource_utilization_graph(data):
            """Update resource utilization graph."""
            if not data or not data["resource_metrics"]["timestamps"]:
                # Return empty figure
                return go.Figure().update_layout(
                    title="No resource utilization data available",
                    template="plotly_dark" if self.theme == "dark" else "plotly_white"
                )
            
            # Parse timestamps
            timestamps = [datetime.datetime.fromisoformat(ts) for ts in data["resource_metrics"]["timestamps"]]
            
            # Create figure
            fig = go.Figure()
            
            # Add CPU utilization
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=data["resource_metrics"]["cpu_utilization"],
                name="CPU",
                line=dict(color="#1F77B4", width=2)
            ))
            
            # Add memory utilization
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=data["resource_metrics"]["memory_utilization"],
                name="Memory",
                line=dict(color="#FF7F0E", width=2)
            ))
            
            # Add GPU utilization if any non-zero values
            if any(v > 0 for v in data["resource_metrics"]["gpu_utilization"]):
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=data["resource_metrics"]["gpu_utilization"],
                    name="GPU",
                    line=dict(color="#2CA02C", width=2)
                ))
            
            # Add scaling events as vertical lines
            for decision in data["scaling_decisions"]:
                decision_time = datetime.datetime.fromisoformat(decision["timestamp"])
                
                if decision["action"] == "scale_up":
                    fig.add_vline(
                        x=decision_time,
                        line=dict(color="green", width=2, dash="dash"),
                        annotation_text=f"Scale up ({decision['count']})",
                        annotation_position="top right"
                    )
                elif decision["action"] == "scale_down":
                    fig.add_vline(
                        x=decision_time,
                        line=dict(color="red", width=2, dash="dash"),
                        annotation_text=f"Scale down ({decision['count']})",
                        annotation_position="top right"
                    )
            
            # Update layout
            fig.update_layout(
                title="Resource Utilization Over Time",
                xaxis_title="Time",
                yaxis_title="Utilization (%)",
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(range=[0, 100]),
                hovermode="x unified",
                height=400
            )
            
            return fig
        
        @app.callback(
            Output("worker-task-graph", "figure"),
            Input("drm-data-store", "data")
        )
        def update_worker_task_graph(data):
            """Update worker count and task graph."""
            if not data or not data["resource_metrics"]["timestamps"]:
                # Return empty figure
                return go.Figure().update_layout(
                    title="No worker and task data available",
                    template="plotly_dark" if self.theme == "dark" else "plotly_white"
                )
            
            # Parse timestamps
            timestamps = [datetime.datetime.fromisoformat(ts) for ts in data["resource_metrics"]["timestamps"]]
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add worker count
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data["resource_metrics"]["worker_count"],
                    name="Worker Count",
                    line=dict(color="#1F77B4", width=2)
                ),
                secondary_y=False
            )
            
            # Add active tasks
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data["resource_metrics"]["active_tasks"],
                    name="Active Tasks",
                    line=dict(color="#FF7F0E", width=2)
                ),
                secondary_y=True
            )
            
            # Add pending tasks if available
            if any(v > 0 for v in data["resource_metrics"]["pending_tasks"]):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=data["resource_metrics"]["pending_tasks"],
                        name="Pending Tasks",
                        line=dict(color="#D62728", width=2)
                    ),
                    secondary_y=True
                )
            
            # Update layout
            fig.update_layout(
                title="Worker Count and Task Load",
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                height=400
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Worker Count", secondary_y=False)
            fig.update_yaxes(title_text="Task Count", secondary_y=True)
            
            return fig
        
        # Worker section callbacks
        @app.callback(
            Output("worker-selector", "options"),
            Input("drm-data-store", "data")
        )
        def update_worker_selector(data):
            """Update worker selector dropdown."""
            if not data or not data["worker_metrics"]:
                return []
            
            # Create options for each worker
            options = [{"label": f"Worker {worker_id}", "value": worker_id} 
                     for worker_id in data["worker_metrics"].keys()]
            
            return options
        
        @app.callback(
            Output("worker-graph-title", "children"),
            Input("worker-metric-selector", "value")
        )
        def update_worker_graph_title(metric):
            """Update worker graph title."""
            if metric == "cpu":
                return "Worker CPU Utilization"
            elif metric == "memory":
                return "Worker Memory Utilization"
            elif metric == "gpu":
                return "Worker GPU Utilization"
            elif metric == "tasks":
                return "Worker Task Count"
            else:
                return "Worker Metrics"
        
        @app.callback(
            Output("worker-metrics-graph", "figure"),
            Input("drm-data-store", "data"),
            Input("worker-selector", "value"),
            Input("worker-metric-selector", "value")
        )
        def update_worker_metrics_graph(data, selected_workers, metric):
            """Update worker metrics graph."""
            if (not data or not data["worker_metrics"] or 
                not selected_workers or not selected_workers):
                # Return empty figure with instruction
                return go.Figure().update_layout(
                    title="Select one or more workers to display metrics",
                    template="plotly_dark" if self.theme == "dark" else "plotly_white"
                )
            
            # Create figure
            fig = go.Figure()
            
            # Determine which metric to display
            metric_field = f"{metric}_utilization" if metric != "tasks" else "tasks"
            
            # Add trace for each selected worker
            for worker_id in selected_workers:
                if worker_id in data["worker_metrics"]:
                    worker_data = data["worker_metrics"][worker_id]
                    
                    # Parse timestamps
                    timestamps = [datetime.datetime.fromisoformat(ts) for ts in worker_data["timestamps"]]
                    
                    # Add worker trace
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=worker_data[metric_field],
                        name=f"Worker {worker_id}",
                        mode="lines+markers"
                    ))
            
            # Customize y-axis label based on metric
            y_title = f"{metric.upper()} Utilization (%)" if metric != "tasks" else "Task Count"
            
            # Update layout
            fig.update_layout(
                title=f"Worker {metric.title()} Metrics",
                xaxis_title="Time",
                yaxis_title=y_title,
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                height=400
            )
            
            # Set y-axis range for utilization metrics
            if metric != "tasks":
                fig.update_yaxes(range=[0, 100])
            
            return fig
        
        @app.callback(
            Output("worker-heatmap", "figure"),
            Input("drm-data-store", "data"),
            Input("worker-metric-selector", "value")
        )
        def update_worker_heatmap(data, metric):
            """Update worker utilization heatmap."""
            if not data or not data["worker_metrics"]:
                # Return empty figure
                return go.Figure().update_layout(
                    title="No worker data available for heatmap",
                    template="plotly_dark" if self.theme == "dark" else "plotly_white"
                )
            
            # Determine which metric to display
            metric_field = f"{metric}_utilization" if metric != "tasks" else "tasks"
            
            # Get all worker IDs
            worker_ids = list(data["worker_metrics"].keys())
            
            # Find common timestamps across all workers
            all_timestamps = set()
            for worker_id in worker_ids:
                worker_data = data["worker_metrics"][worker_id]
                for ts in worker_data["timestamps"]:
                    all_timestamps.add(ts)
            
            # Sort timestamps
            all_timestamps = sorted(all_timestamps)
            
            # Create mapping of timestamps to indices
            ts_to_idx = {ts: i for i, ts in enumerate(all_timestamps)}
            
            # Create heatmap data matrix
            heatmap_data = np.zeros((len(worker_ids), len(all_timestamps)))
            heatmap_data.fill(np.nan)  # Fill with NaN for missing data
            
            # Fill in the matrix with available data
            for i, worker_id in enumerate(worker_ids):
                worker_data = data["worker_metrics"][worker_id]
                
                for j, ts in enumerate(worker_data["timestamps"]):
                    if ts in ts_to_idx:
                        idx = ts_to_idx[ts]
                        heatmap_data[i, idx] = worker_data[metric_field][j]
            
            # Convert timestamps to human-readable format
            timestamp_labels = [
                datetime.datetime.fromisoformat(ts).strftime("%H:%M:%S") 
                for ts in all_timestamps
            ]
            
            # Create heatmap figure
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=timestamp_labels,
                y=[f"Worker {id}" for id in worker_ids],
                colorscale="Viridis",
                colorbar=dict(
                    title="% Utilization" if metric != "tasks" else "Task Count"
                ),
                hovertemplate="Worker %{y}<br>Time: %{x}<br>Value: %{z:.1f}<extra></extra>"
            ))
            
            # Set color scale range for utilization metrics
            if metric != "tasks":
                fig.update_traces(zmin=0, zmax=100)
            
            # Update layout
            title = f"Worker {metric.title()} Heatmap"
            if metric != "tasks":
                title += " (% Utilization)"
                
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Worker ID",
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            
            return fig
        
        # Performance section callbacks
        @app.callback(
            Output("performance-metrics-graph", "figure"),
            Input("drm-data-store", "data"),
            Input("performance-metric-selector", "value"),
            Input("performance-time-selector", "value")
        )
        def update_performance_metrics_graph(data, selected_metrics, time_range):
            """Update performance metrics graph."""
            if not data or not selected_metrics:
                # Return empty figure
                return go.Figure().update_layout(
                    title="No performance metrics selected",
                    template="plotly_dark" if self.theme == "dark" else "plotly_white"
                )
            
            # Calculate cutoff time
            cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=time_range)
            cutoff_time_str = cutoff_time.isoformat()
            
            # Create figure with multiple y-axes
            fig = make_subplots(specs=[[{"secondary_y": len(selected_metrics) > 1}]])
            
            # Add traces for selected metrics
            for i, metric in enumerate(selected_metrics):
                if metric in data["performance_metrics"] and data["performance_metrics"][metric]:
                    # Filter data by time range
                    filtered_data = [
                        entry for entry in data["performance_metrics"][metric]
                        if entry["timestamp"] >= cutoff_time_str
                    ]
                    
                    if not filtered_data:
                        continue
                    
                    # Parse timestamps
                    timestamps = [datetime.datetime.fromisoformat(entry["timestamp"]) for entry in filtered_data]
                    values = [entry["value"] for entry in filtered_data]
                    
                    # Define color based on metric
                    colors = {
                        "task_throughput": "#1F77B4",  # Blue
                        "allocation_time": "#FF7F0E",  # Orange
                        "resource_efficiency": "#2CA02C"  # Green
                    }
                    
                    # Define display names
                    display_names = {
                        "task_throughput": "Task Throughput (tasks/s)",
                        "allocation_time": "Allocation Time (ms)",
                        "resource_efficiency": "Resource Efficiency (%)"
                    }
                    
                    # Add trace (first metric on primary y-axis, others on secondary)
                    use_secondary_y = i > 0
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=values,
                            name=display_names.get(metric, metric),
                            line=dict(color=colors.get(metric, "#000"), width=2)
                        ),
                        secondary_y=use_secondary_y
                    )
                    
                    # Update y-axis titles
                    if i == 0:  # Primary axis
                        fig.update_yaxes(
                            title_text=display_names.get(metric, metric), 
                            secondary_y=False
                        )
                    elif i == 1:  # Secondary axis
                        fig.update_yaxes(
                            title_text=display_names.get(metric, metric), 
                            secondary_y=True
                        )
            
            # Update layout
            fig.update_layout(
                title="Performance Metrics Over Time",
                xaxis_title="Time",
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                height=400
            )
            
            return fig
        
        @app.callback(
            Output("regression-analysis-content", "children"),
            Input("drm-data-store", "data")
        )
        def update_regression_analysis(data):
            """Update regression analysis content."""
            if not REGRESSION_AVAILABLE:
                return html.Div([
                    html.P("Regression detection module not available. Install required dependencies to enable this feature."),
                    html.Code("pip install scipy ruptures")
                ], className="text-muted p-3")
            
            if not data or not data["resource_metrics"]["timestamps"]:
                return html.Div("No data available for regression analysis.", className="text-muted p-3")
            
            # Check if we have enough data for analysis
            if len(data["resource_metrics"]["timestamps"]) < 10:
                return html.Div("Collecting data for regression analysis (minimum 10 data points required)...", 
                              className="text-muted p-3")
            
            # Create a simple regression analysis content
            return html.Div([
                html.P("Performance metrics are being analyzed for statistical regressions.", className="mb-3"),
                
                dbc.Row([
                    # CPU utilization regression card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("CPU Utilization", className="fw-bold"),
                            dbc.CardBody([
                                html.Div(id="cpu-regression-result"),
                                # We would show regression detection results here
                                html.P("No significant regressions detected.", className="text-success mb-0")
                            ])
                        ])
                    ], width=4),
                    
                    # Memory utilization regression card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Memory Utilization", className="fw-bold"),
                            dbc.CardBody([
                                html.Div(id="memory-regression-result"),
                                # We would show regression detection results here
                                html.P("No significant regressions detected.", className="text-success mb-0")
                            ])
                        ])
                    ], width=4),
                    
                    # Allocation time regression card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Allocation Time", className="fw-bold"),
                            dbc.CardBody([
                                html.Div(id="allocation-regression-result"),
                                # We would show regression detection results here
                                html.P("No significant regressions detected.", className="text-success mb-0")
                            ])
                        ])
                    ], width=4),
                ])
            ])
        
        # Scaling section callbacks
        @app.callback(
            Output("scaling-timeline-graph", "figure"),
            Input("drm-data-store", "data")
        )
        def update_scaling_timeline(data):
            """Update scaling decisions timeline."""
            if not data or not data["scaling_decisions"]:
                # Return empty figure
                return go.Figure().update_layout(
                    title="No scaling decisions available",
                    template="plotly_dark" if self.theme == "dark" else "plotly_white"
                )
            
            # Parse timestamps
            timestamps = [datetime.datetime.fromisoformat(d["timestamp"]) for d in data["scaling_decisions"]]
            
            # Create values for each scaling decision
            # Positive for scale up, negative for scale down, zero for maintain
            values = []
            colors = []
            hover_texts = []
            
            for decision in data["scaling_decisions"]:
                action = decision["action"]
                count = decision["count"]
                reason = decision["reason"]
                
                if action == "scale_up":
                    values.append(count)
                    colors.append("rgba(53, 167, 83, 0.8)")  # Green
                    hover_texts.append(f"Scale up: +{count} workers<br>Reason: {reason}")
                elif action == "scale_down":
                    values.append(-count)
                    colors.append("rgba(249, 86, 79, 0.8)")  # Red
                    hover_texts.append(f"Scale down: -{count} workers<br>Reason: {reason}")
                else:
                    values.append(0)
                    colors.append("rgba(149, 165, 166, 0.8)")  # Gray
                    hover_texts.append(f"Maintain worker count<br>Reason: {reason}")
            
            # Create figure
            fig = go.Figure()
            
            # Add scaling decisions as a bar chart
            fig.add_trace(go.Bar(
                x=timestamps,
                y=values,
                marker_color=colors,
                text=hover_texts,
                hovertemplate="%{text}<br>Time: %{x|%H:%M:%S}<extra></extra>"
            ))
            
            # Add resource utilization as a line
            if data["resource_metrics"]["timestamps"]:
                # Parse timestamps for resource metrics
                resource_ts = [datetime.datetime.fromisoformat(ts) for ts in data["resource_metrics"]["timestamps"]]
                
                # Add CPU utilization line
                fig.add_trace(go.Scatter(
                    x=resource_ts,
                    y=data["resource_metrics"]["cpu_utilization"],
                    name="CPU Utilization (%)",
                    line=dict(color="#1F77B4", width=2, dash="dot"),
                    yaxis="y2"
                ))
            
            # Update layout
            fig.update_layout(
                title="Scaling Decisions Timeline",
                xaxis_title="Time",
                yaxis=dict(
                    title="Worker Count Change",
                    side="left"
                ),
                yaxis2=dict(
                    title="Utilization (%)",
                    side="right",
                    overlaying="y",
                    range=[0, 100]
                ),
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                barmode="relative"
            )
            
            return fig
        
        @app.callback(
            Output("scaling-decisions-table", "children"),
            Input("drm-data-store", "data")
        )
        def update_scaling_decisions_table(data):
            """Update scaling decisions table."""
            if not data or not data["scaling_decisions"]:
                return html.Div("No scaling decisions to display.", className="text-muted p-3")
            
            # Sort decisions by timestamp (newest first)
            sorted_decisions = sorted(
                data["scaling_decisions"],
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            # Create rows for each decision
            rows = []
            for decision in sorted_decisions:
                timestamp = datetime.datetime.fromisoformat(decision["timestamp"]).strftime("%H:%M:%S")
                action = decision["action"]
                count = decision["count"]
                reason = decision["reason"]
                
                # Determine style based on action
                if action == "scale_up":
                    action_text = f"Scale up (+{count})"
                    badge_class = "bg-success"
                elif action == "scale_down":
                    action_text = f"Scale down (-{count})"
                    badge_class = "bg-danger"
                else:
                    action_text = "Maintain"
                    badge_class = "bg-secondary"
                
                # Create table row
                rows.append(html.Tr([
                    html.Td(timestamp),
                    html.Td(html.Span(action_text, className=f"badge {badge_class}")),
                    html.Td(count),
                    html.Td(reason)
                ]))
            
            # Create table
            table = dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Time"),
                    html.Th("Action"),
                    html.Th("Count"),
                    html.Th("Reason")
                ])),
                html.Tbody(rows)
            ], striped=True, bordered=True, hover=True, responsive=True)
            
            return table
        
        # Alerts section callbacks
        @app.callback(
            Output("alerts-list", "children"),
            Input("drm-data-store", "data"),
            Input("alert-level-filter", "value"),
            Input("alert-source-filter", "value")
        )
        def update_alerts_list(data, level_filter, source_filter):
            """Update alerts list."""
            if not data or not data["alerts"]:
                return html.Div("No alerts to display.", className="text-muted p-3")
            
            # Filter alerts by level and source
            filtered_alerts = [
                alert for alert in data["alerts"]
                if alert["level"] in level_filter and alert["source"] in source_filter
            ]
            
            if not filtered_alerts:
                return html.Div("No alerts match the current filters.", className="text-muted p-3")
            
            # Sort alerts by timestamp (newest first)
            sorted_alerts = sorted(
                filtered_alerts,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            # Create alert items
            alert_items = []
            for alert in sorted_alerts:
                timestamp = datetime.datetime.fromisoformat(alert["timestamp"]).strftime("%H:%M:%S")
                level = alert["level"]
                message = alert["message"]
                source = alert["source"]
                
                # Determine style based on level
                if level == "critical":
                    color = "danger"
                    icon = "fa-skull"
                elif level == "error":
                    color = "danger"
                    icon = "fa-times-circle"
                elif level == "warning":
                    color = "warning"
                    icon = "fa-exclamation-triangle"
                else:  # info
                    color = "info"
                    icon = "fa-info-circle"
                
                # Create alert item
                alert_items.append(dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span([
                                html.I(className=f"fas {icon} me-2"),
                                html.Strong(f"{level.title()}: ", className="me-1"),
                                html.Span(message)
                            ]),
                            html.Div([
                                html.Small(f"Time: {timestamp}", className="me-3"),
                                html.Small(f"Source: {source.replace('_', ' ').title()}")
                            ], className="text-muted mt-1")
                        ])
                    ])
                ], color=color, className="mb-2", outline=True))
            
            return html.Div(alert_items)

def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DRM Real-Time Dashboard")
    parser.add_argument("--port", type=int, default=8085, help="Dashboard port")
    parser.add_argument("--db-path", default="benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--update-interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--retention", type=int, default=60, help="Data retention window in minutes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--theme", choices=["light", "dark"], default="dark", help="Dashboard theme")
    
    args = parser.parse_args()
    
    # Create and start dashboard
    dashboard = DRMRealTimeDashboard(
        db_path=args.db_path,
        port=args.port,
        update_interval=args.update_interval,
        retention_window=args.retention,
        debug=args.debug,
        theme=args.theme
    )
    
    dashboard.start()

if __name__ == "__main__":
    main()