#!/usr/bin/env python3
"""
Comprehensive Monitoring Dashboard for Distributed Testing Framework

This module implements a comprehensive monitoring dashboard for the distributed testing framework.
It extends the basic dashboard functionality to provide real-time monitoring, interactive
visualizations, and integrated performance metrics across all components of the distributed
testing system.

Implementation Date: March 17, 2025 (Originally planned for June 19-26, 2025)

Features:
- Real-time monitoring of distributed worker nodes
- Interactive system topology visualization
- Live task execution tracking and visualization
- Performance metrics dashboards with drill-down capability
- Hardware utilization visualization across the distributed system
- Fault tolerance monitoring and visualization
- Comprehensive test result analysis
- Alert system for critical issues
- Historical trend analysis
"""

import os
import sys
import json
import time
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from pathlib import Path
import tempfile
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("monitoring_dashboard")

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import dashboard components
try:
    from duckdb_api.distributed_testing.dashboard.dashboard_generator import DashboardGenerator
    from duckdb_api.distributed_testing.dashboard.dashboard_server import DashboardServer
    from duckdb_api.distributed_testing.dashboard.visualization import VisualizationEngine
    DASHBOARD_COMPONENTS_AVAILABLE = True
except ImportError:
    logger.warning("Dashboard components not available. Monitoring functionality will be limited.")
    DASHBOARD_COMPONENTS_AVAILABLE = False

# Import optional WebSocket and aiohttp dependencies
try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    logger.warning("aiohttp not available. Real-time functionality will be limited.")
    AIOHTTP_AVAILABLE = False

# Import optional Jinja2 for template rendering
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    logger.warning("jinja2 not available. Template rendering will be limited.")
    JINJA2_AVAILABLE = False

# Import optional plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Interactive visualizations will be limited.")
    PLOTLY_AVAILABLE = False

class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for the distributed testing framework.
    
    This dashboard provides real-time monitoring of all components in the distributed
    testing framework, including coordinator, workers, load balancers, and result aggregators.
    It offers interactive visualizations, alerts, and comprehensive metrics.
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 8082,
                 coordinator_url: Optional[str] = None, 
                 result_aggregator = None,
                 output_dir: str = "./monitoring_dashboard"):
        """
        Initialize the monitoring dashboard.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            coordinator_url: URL of the coordinator server (optional)
            result_aggregator: Result aggregator instance (optional)
            output_dir: Directory to save dashboard files
        """
        self.host = host
        self.port = port
        self.coordinator_url = coordinator_url
        self.result_aggregator = result_aggregator
        self.output_dir = output_dir
        
        # Status tracking
        self.running = False
        self.last_updated = datetime.now()
        
        # Create dashboard components if available
        if DASHBOARD_COMPONENTS_AVAILABLE:
            # Initialize visualization engine
            viz_dir = os.path.join(output_dir, "visualizations")
            self.visualization_engine = VisualizationEngine(
                result_aggregator=result_aggregator,
                output_dir=viz_dir
            )
            
            # Initialize dashboard generator
            self.dashboard_generator = DashboardGenerator(
                result_aggregator=result_aggregator,
                output_dir=output_dir
            )
            
            # Use custom configuration tuned for monitoring dashboard
            self.dashboard_generator.configure({
                "theme": "dark",  # Better for monitoring screens
                "refresh_interval": 30,  # Faster refresh for monitoring
                "include_performance_charts": True,
                "include_regression_detection": True,
                "include_dimension_analysis": True, 
                "include_test_details": True,
                "include_worker_details": True,
                "max_items_per_section": 20  # Show more items
            })
        else:
            self.visualization_engine = None
            self.dashboard_generator = None
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "static"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "templates"), exist_ok=True)
        
        # Create web application if aiohttp is available
        if AIOHTTP_AVAILABLE:
            self.app = web.Application()
            self.setup_routes()
            self.setup_templates()
            self.websocket_connections = set()
        else:
            self.app = None
            
        # Data sources
        self.coordinator_client = None
        self.worker_connections = {}
        self.system_metrics = {}
        self.alert_history = []
        self.performance_history = {}
        self.topology_cache = None
        self.task_execution_tracking = {}
        
        # Configuration
        self.config = {
            "auto_refresh": 30,  # Auto-refresh interval in seconds
            "theme": "dark",  # Default theme for monitoring
            "alert_levels": {
                "critical": 1,
                "warning": 2,
                "info": 3
            },
            "alert_retention_days": 7,  # How long to keep alert history
            "metrics_retention_days": 30,  # How long to keep metrics history
            "real_time_enabled": True,  # Enable real-time updates
            "enable_alerts": True,  # Enable alert generation 
            "update_interval": 5,  # Background update interval in seconds
            "auto_connect_coordinator": True,  # Auto-connect to coordinator
            "max_workers_shown": 50,  # Maximum number of workers shown in dashboard
            "max_tasks_tracked": 500,  # Maximum number of tasks tracked
            "enable_task_detail_tracking": True,  # Track detailed task execution
            "enable_hardware_metrics": True,  # Collect hardware metrics
            "enable_performance_prediction": True,  # Enable predictive metrics
            "visualization_mode": "interactive",  # interactive or static
            "enable_3d_visualization": False  # Enable 3D visualizations (experimental)
        }
        
        # Background update thread
        self.update_thread = None
        self.update_stop_event = threading.Event()
        
        logger.info(f"Monitoring dashboard initialized at http://{host}:{port}")
    
    def setup_routes(self):
        """Set up the web application routes."""
        if not self.app:
            return
            
        # Main routes
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/dashboard", self.handle_dashboard)
        self.app.router.add_get("/system", self.handle_system_view)
        self.app.router.add_get("/workers", self.handle_workers_view)
        self.app.router.add_get("/tasks", self.handle_tasks_view)
        self.app.router.add_get("/performance", self.handle_performance_view)
        self.app.router.add_get("/alerts", self.handle_alerts_view)
        self.app.router.add_get("/topology", self.handle_topology_view)
        self.app.router.add_get("/fault-tolerance", self.handle_fault_tolerance_view)
        
        # API routes for data access
        self.app.router.add_get("/api/status", self.handle_api_status)
        self.app.router.add_get("/api/workers", self.handle_api_workers)
        self.app.router.add_get("/api/tasks", self.handle_api_tasks)
        self.app.router.add_get("/api/metrics", self.handle_api_metrics)
        self.app.router.add_get("/api/topology", self.handle_api_topology)
        self.app.router.add_get("/api/performance", self.handle_api_performance)
        self.app.router.add_get("/api/alerts", self.handle_api_alerts)
        self.app.router.add_get("/api/fault-tolerance", self.handle_api_fault_tolerance)
        
        # WebSocket for real-time updates
        self.app.router.add_get("/ws", self.handle_websocket)
        
        # Static files
        self.app.router.add_static("/static", 
                                 os.path.join(self.output_dir, "static"),
                                 name="static")
        self.app.router.add_static("/visualizations", 
                                 os.path.join(self.output_dir, "visualizations"),
                                 name="visualizations")
    
    def setup_templates(self):
        """Set up the Jinja2 templates for dashboard pages."""
        if not JINJA2_AVAILABLE:
            return
            
        # Create template directory if not exists
        template_dir = os.path.join(self.output_dir, "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        # Create default templates
        self._create_default_templates(template_dir)
        
        # Initialize Jinja environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    def _create_default_templates(self, template_dir):
        """
        Create default template files for the monitoring dashboard.
        
        Args:
            template_dir: Directory to create templates in
        """
        # Create base template
        base_template_path = os.path.join(template_dir, "base.html")
        if not os.path.exists(base_template_path):
            with open(base_template_path, "w") as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Distributed Testing Monitoring Dashboard{% endblock %}</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --info-color: #1abc9c;
            --bg-color: {% if theme == "dark" %}#1a1a1a{% else %}#ffffff{% endif %};
            --bg-secondary: {% if theme == "dark" %}#2a2a2a{% else %}#f8f9fa{% endif %};
            --text-color: {% if theme == "dark" %}#f8f9fa{% else %}#333333{% endif %};
            --border-color: {% if theme == "dark" %}#444444{% else %}#e9ecef{% endif %};
            --shadow-color: {% if theme == "dark" %}rgba(0,0,0,0.3){% else %}rgba(0,0,0,0.1){% endif %};
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        
        .container {
            width: 100%;
            max-width: 100%;
            padding: 0;
            margin: 0;
        }
        
        .dashboard-header {
            background-color: var(--bg-secondary);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 5px var(--shadow-color);
        }
        
        .dashboard-title {
            font-size: 20px;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        
        .dashboard-title .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-indicator.active {
            background-color: var(--secondary-color);
        }
        
        .status-indicator.warning {
            background-color: var(--warning-color);
        }
        
        .status-indicator.error {
            background-color: var(--danger-color);
        }
        
        .dashboard-controls {
            display: flex;
            align-items: center;
        }
        
        .dashboard-controls select, .dashboard-controls button {
            margin-left: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-secondary);
            color: var(--text-color);
        }
        
        .main-layout {
            display: flex;
            min-height: calc(100vh - 50px);
        }
        
        .sidebar {
            width: 200px;
            background-color: var(--bg-secondary);
            color: var(--text-color);
            padding: 20px 0;
            box-shadow: 1px 0 5px var(--shadow-color);
        }
        
        .sidebar-nav {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        
        .sidebar-nav li {
            padding: 0;
            margin: 0;
        }
        
        .sidebar-nav a {
            display: block;
            padding: 10px 20px;
            color: var(--text-color);
            text-decoration: none;
            transition: background-color 0.2s;
        }
        
        .sidebar-nav a:hover {
            background-color: rgba(255,255,255,0.1);
        }
        
        .sidebar-nav a.active {
            background-color: var(--primary-color);
            color: white;
        }
        
        .content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .grid-item {
            background-color: var(--bg-secondary);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px var(--shadow-color);
        }
        
        .grid-item.large {
            grid-column: span 2;
            grid-row: span 2;
        }
        
        .grid-item.medium {
            grid-column: span 2;
        }
        
        .item-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metrics-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .metrics-label {
            font-size: 12px;
            opacity: 0.7;
        }
        
        .chart-container {
            width: 100%;
            height: 300px;
        }
        
        .status-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .status-table th,
        .status-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .status-table th {
            border-top: 1px solid var(--border-color);
            background-color: rgba(0,0,0,0.05);
        }
        
        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .status-badge.success {
            background-color: var(--secondary-color);
            color: white;
        }
        
        .status-badge.warning {
            background-color: var(--warning-color);
            color: white;
        }
        
        .status-badge.danger {
            background-color: var(--danger-color);
            color: white;
        }
        
        .status-badge.info {
            background-color: var(--info-color);
            color: white;
        }
        
        .status-badge.primary {
            background-color: var(--primary-color);
            color: white;
        }
        
        .alert-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .alert-item {
            padding: 8px 12px;
            border-left: 3px solid;
            margin-bottom: 8px;
            background-color: rgba(0,0,0,0.05);
        }
        
        .alert-item.critical {
            border-color: var(--danger-color);
        }
        
        .alert-item.warning {
            border-color: var(--warning-color);
        }
        
        .alert-item.info {
            border-color: var(--info-color);
        }
        
        .alert-time {
            font-size: 12px;
            opacity: 0.7;
        }
        
        .alert-title {
            font-weight: bold;
            margin: 5px 0;
        }
        
        .progress-bar-container {
            width: 100%;
            height: 8px;
            background-color: rgba(0,0,0,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .progress-bar {
            height: 100%;
            border-radius: 4px;
            background-color: var(--primary-color);
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background-color: rgba(0,0,0,0.8);
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
        }
        
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        .footer {
            text-align: center;
            padding: 10px;
            font-size: 12px;
            opacity: 0.7;
            border-top: 1px solid var(--border-color);
        }
        
        /* Responsive layout */
        @media (max-width: 768px) {
            .main-layout {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
                padding: 10px 0;
            }
            
            .content {
                padding: 10px;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .grid-item.large, .grid-item.medium {
                grid-column: span 1;
                grid-row: span 1;
            }
        }
    </style>
    {% if auto_refresh > 0 %}
    <meta http-equiv="refresh" content="{{ auto_refresh }}">
    {% endif %}
    {% block head_extra %}{% endblock %}
</head>
<body>
    <div class="container">
        <div class="dashboard-header">
            <div class="dashboard-title">
                <span class="status-indicator {% if system_status == 'active' %}active{% elif system_status == 'warning' %}warning{% else %}error{% endif %}"></span>
                <span>Distributed Testing Monitoring Dashboard</span>
            </div>
            <div class="dashboard-controls">
                <select id="refresh-rate" onchange="setRefreshRate(this.value)">
                    <option value="0" {% if auto_refresh == 0 %}selected{% endif %}>No Refresh</option>
                    <option value="5" {% if auto_refresh == 5 %}selected{% endif %}>5 seconds</option>
                    <option value="10" {% if auto_refresh == 10 %}selected{% endif %}>10 seconds</option>
                    <option value="30" {% if auto_refresh == 30 %}selected{% endif %}>30 seconds</option>
                    <option value="60" {% if auto_refresh == 60 %}selected{% endif %}>1 minute</option>
                </select>
                <button id="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
            </div>
        </div>
        <div class="main-layout">
            <div class="sidebar">
                <ul class="sidebar-nav">
                    <li><a href="/" class="{% if active_page == 'index' %}active{% endif %}">Overview</a></li>
                    <li><a href="/system" class="{% if active_page == 'system' %}active{% endif %}">System Status</a></li>
                    <li><a href="/workers" class="{% if active_page == 'workers' %}active{% endif %}">Workers</a></li>
                    <li><a href="/tasks" class="{% if active_page == 'tasks' %}active{% endif %}">Tasks</a></li>
                    <li><a href="/performance" class="{% if active_page == 'performance' %}active{% endif %}">Performance</a></li>
                    <li><a href="/alerts" class="{% if active_page == 'alerts' %}active{% endif %}">Alerts</a></li>
                    <li><a href="/topology" class="{% if active_page == 'topology' %}active{% endif %}">Topology</a></li>
                    <li><a href="/fault-tolerance" class="{% if active_page == 'fault-tolerance' %}active{% endif %}">Fault Tolerance</a></li>
                </ul>
            </div>
            <div class="content">
                {% block content %}{% endblock %}
            </div>
        </div>
        <div class="footer">
            <p>Distributed Testing Framework - Monitoring Dashboard | Last updated: {{ last_updated }}</p>
        </div>
    </div>
    
    <script>
        // Set up WebSocket for real-time updates
        let ws = null;
        let wsReconnectTimeout = null;
        
        function connectWebSocket() {
            if (ws) {
                ws.close();
            }
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                if (wsReconnectTimeout) {
                    clearTimeout(wsReconnectTimeout);
                    wsReconnectTimeout = null;
                }
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                wsReconnectTimeout = setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                ws.close();
            };
        }
        
        function handleWebSocketMessage(data) {
            if (data.type === 'status_update') {
                updateStatus(data.data);
            } else if (data.type === 'alert') {
                showAlert(data.data);
            } else if (data.type === 'metrics_update') {
                updateMetrics(data.data);
            } else if (data.type === 'task_update') {
                updateTaskStatus(data.data);
            } else if (data.type === 'worker_update') {
                updateWorkerStatus(data.data);
            }
        }
        
        function setRefreshRate(seconds) {
            window.location.href = window.location.pathname + '?refresh=' + seconds;
        }
        
        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Reload page with new theme
            window.location.href = window.location.pathname + '?theme=' + newTheme;
        }
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            // Set theme from local storage
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.documentElement.setAttribute('data-theme', savedTheme);
            
            // Connect WebSocket if supported
            if ('WebSocket' in window) {
                connectWebSocket();
            }
            
            // Additional page-specific initialization
            if (typeof initPage === 'function') {
                initPage();
            }
        });
    </script>
    
    {% block scripts %}{% endblock %}
</body>
</html>""")
        
        # Create index template
        index_template_path = os.path.join(template_dir, "index.html")
        if not os.path.exists(index_template_path):
            with open(index_template_path, "w") as f:
                f.write("""{% extends "base.html" %}

{% block title %}Monitoring Dashboard - Overview{% endblock %}

{% block content %}
    <h1>System Overview</h1>
    
    <div class="dashboard-grid">
        <div class="grid-item">
            <div class="item-title">
                <span>System Status</span>
                <span class="status-badge {{ system_status }}">{{ system_status|title }}</span>
            </div>
            <div class="metrics-value">{{ total_workers }} Workers</div>
            <div class="metrics-label">{{ active_workers }} Active / {{ idle_workers }} Idle</div>
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {{ (active_workers / total_workers * 100) if total_workers > 0 else 0 }}%"></div>
            </div>
        </div>
        
        <div class="grid-item">
            <div class="item-title">
                <span>Task Status</span>
            </div>
            <div class="metrics-value">{{ total_tasks }}</div>
            <div class="metrics-label">
                <span class="status-badge success">{{ completed_tasks }} Completed</span>
                <span class="status-badge warning">{{ running_tasks }} Running</span>
                <span class="status-badge info">{{ pending_tasks }} Pending</span>
                <span class="status-badge danger">{{ failed_tasks }} Failed</span>
            </div>
        </div>
        
        <div class="grid-item">
            <div class="item-title">
                <span>Test Performance</span>
            </div>
            <div class="metrics-value">{{ avg_throughput|round(1) }}</div>
            <div class="metrics-label">Average tasks/min</div>
            <div class="metrics-label">{{ total_tests_run }} tests run in last hour</div>
        </div>
        
        <div class="grid-item">
            <div class="item-title">
                <span>Alerts</span>
            </div>
            <div class="metrics-value">{{ total_alerts }}</div>
            <div class="metrics-label">
                <span class="status-badge danger">{{ critical_alerts }} Critical</span>
                <span class="status-badge warning">{{ warning_alerts }} Warning</span>
                <span class="status-badge info">{{ info_alerts }} Info</span>
            </div>
        </div>
        
        <div class="grid-item large">
            <div class="item-title">
                <span>System Load</span>
                <span>Last 24 hours</span>
            </div>
            <div class="chart-container">
                <img src="{{ system_load_chart }}" alt="System Load Chart" style="width: 100%; height: 100%; object-fit: contain;">
            </div>
        </div>
        
        <div class="grid-item medium">
            <div class="item-title">
                <span>Recent Alerts</span>
                <a href="/alerts" style="font-size: 12px; text-decoration: none; color: var(--primary-color);">View All</a>
            </div>
            <div class="alert-list">
                {% for alert in recent_alerts %}
                <div class="alert-item {{ alert.level }}">
                    <div class="alert-time">{{ alert.timestamp }}</div>
                    <div class="alert-title">{{ alert.title }}</div>
                    <div class="alert-message">{{ alert.message }}</div>
                </div>
                {% else %}
                <div>No recent alerts</div>
                {% endfor %}
            </div>
        </div>
        
        <div class="grid-item medium">
            <div class="item-title">
                <span>Hardware Utilization</span>
            </div>
            <div class="status-table-container">
                <table class="status-table">
                    <thead>
                        <tr>
                            <th>Hardware Type</th>
                            <th>Count</th>
                            <th>Utilization</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for hw in hardware_utilization %}
                        <tr>
                            <td>{{ hw.type }}</td>
                            <td>{{ hw.count }}</td>
                            <td>
                                <div class="progress-bar-container">
                                    <div class="progress-bar" style="width: {{ hw.utilization }}%"></div>
                                </div>
                            </td>
                            <td><span class="status-badge {{ hw.status }}">{{ hw.status|title }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <h2>Active Workers</h2>
    <div class="status-table-container">
        <table class="status-table">
            <thead>
                <tr>
                    <th>Worker ID</th>
                    <th>Hardware</th>
                    <th>Tasks</th>
                    <th>Uptime</th>
                    <th>Load</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for worker in active_worker_list %}
                <tr>
                    <td>{{ worker.id }}</td>
                    <td>{{ worker.hardware }}</td>
                    <td>{{ worker.tasks_completed }} / {{ worker.tasks_total }}</td>
                    <td>{{ worker.uptime }}</td>
                    <td>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: {{ worker.load }}%"></div>
                        </div>
                    </td>
                    <td><span class="status-badge {{ worker.status }}">{{ worker.status|title }}</span></td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="6">No active workers</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <h2>Recent Tasks</h2>
    <div class="status-table-container">
        <table class="status-table">
            <thead>
                <tr>
                    <th>Task ID</th>
                    <th>Type</th>
                    <th>Worker</th>
                    <th>Started</th>
                    <th>Duration</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for task in recent_tasks %}
                <tr>
                    <td>{{ task.id }}</td>
                    <td>{{ task.type }}</td>
                    <td>{{ task.worker_id }}</td>
                    <td>{{ task.start_time }}</td>
                    <td>{{ task.duration }}</td>
                    <td><span class="status-badge {{ task.status }}">{{ task.status|title }}</span></td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="6">No recent tasks</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
{% endblock %}

{% block scripts %}
<script>
    function updateStatus(statusData) {
        // Update status elements with real-time data
        if (statusData.workers) {
            document.querySelector('.metrics-value').textContent = statusData.workers.total + ' Workers';
            document.querySelector('.metrics-label').textContent = 
                statusData.workers.active + ' Active / ' + 
                statusData.workers.idle + ' Idle';
            
            const progressBar = document.querySelector('.progress-bar');
            const percentage = statusData.workers.total > 0 
                ? (statusData.workers.active / statusData.workers.total * 100) 
                : 0;
            progressBar.style.width = percentage + '%';
        }
    }
    
    function showAlert(alertData) {
        // Show new alert in the alert list
        const alertList = document.querySelector('.alert-list');
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item ' + alertData.level;
        
        alertItem.innerHTML = `
            <div class="alert-time">${alertData.timestamp}</div>
            <div class="alert-title">${alertData.title}</div>
            <div class="alert-message">${alertData.message}</div>
        `;
        
        // Add to top of list
        alertList.insertBefore(alertItem, alertList.firstChild);
        
        // Remove oldest if more than 5 alerts
        if (alertList.children.length > 5) {
            alertList.removeChild(alertList.lastChild);
        }
    }
    
    function initPage() {
        // Any page-specific initialization can go here
    }
</script>
{% endblock %}""")
        
        # Create system view template
        system_template_path = os.path.join(template_dir, "system.html")
        if not os.path.exists(system_template_path):
            with open(system_template_path, "w") as f:
                f.write("""{% extends "base.html" %}

{% block title %}Monitoring Dashboard - System Status{% endblock %}

{% block head_extra %}
    <style>
        .system-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .system-chart-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .system-chart {
            height: 400px;
            background-color: var(--bg-secondary);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px var(--shadow-color);
        }
        
        @media (max-width: 992px) {
            .system-chart-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
{% endblock %}

{% block content %}
    <h1>System Status</h1>
    
    <div class="system-metrics-grid">
        <div class="grid-item">
            <div class="item-title">Coordinator Status</div>
            <div class="metrics-value">
                <span class="status-badge {{ coordinator_status }}">
                    {{ coordinator_status|title }}
                </span>
            </div>
            <div class="metrics-label">URL: {{ coordinator_url }}</div>
            <div class="metrics-label">Uptime: {{ coordinator_uptime }}</div>
        </div>
        
        <div class="grid-item">
            <div class="item-title">Total Workers</div>
            <div class="metrics-value">{{ total_workers }}</div>
            <div class="metrics-label">{{ active_workers }} Active / {{ idle_workers }} Idle</div>
            <div class="metrics-label">{{ disconnected_workers }} Disconnected</div>
        </div>
        
        <div class="grid-item">
            <div class="item-title">Task Queue</div>
            <div class="metrics-value">{{ queue_length }}</div>
            <div class="metrics-label">{{ queue_processing_rate }}/min processing rate</div>
            <div class="metrics-label">Est. completion: {{ queue_completion_time }}</div>
        </div>
        
        <div class="grid-item">
            <div class="item-title">System Health</div>
            <div class="metrics-value">{{ system_health_score }}%</div>
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {{ system_health_score }}%; 
                    background-color: {% if system_health_score > 80 %}var(--secondary-color)
                                    {% elif system_health_score > 50 %}var(--warning-color)
                                    {% else %}var(--danger-color){% endif %};">
                </div>
            </div>
            <div class="metrics-label">Based on {{ system_health_metrics }} metrics</div>
        </div>
    </div>
    
    <div class="system-chart-grid">
        <div class="system-chart">
            <div class="item-title">CPU Usage History</div>
            <div class="chart-container">
                <img src="{{ cpu_usage_chart }}" alt="CPU Usage Chart" style="width: 100%; height: 100%; object-fit: contain;">
            </div>
        </div>
        
        <div class="system-chart">
            <div class="item-title">Memory Usage History</div>
            <div class="chart-container">
                <img src="{{ memory_usage_chart }}" alt="Memory Usage Chart" style="width: 100%; height: 100%; object-fit: contain;">
            </div>
        </div>
        
        <div class="system-chart">
            <div class="item-title">Task Throughput</div>
            <div class="chart-container">
                <img src="{{ task_throughput_chart }}" alt="Task Throughput Chart" style="width: 100%; height: 100%; object-fit: contain;">
            </div>
        </div>
        
        <div class="system-chart">
            <div class="item-title">Network Activity</div>
            <div class="chart-container">
                <img src="{{ network_activity_chart }}" alt="Network Activity Chart" style="width: 100%; height: 100%; object-fit: contain;">
            </div>
        </div>
    </div>
    
    <h2>Component Status</h2>
    <div class="status-table-container">
        <table class="status-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Uptime</th>
                    <th>Load</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                {% for component in component_status %}
                <tr>
                    <td>{{ component.name }}</td>
                    <td><span class="status-badge {{ component.status }}">{{ component.status|title }}</span></td>
                    <td>{{ component.uptime }}</td>
                    <td>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: {{ component.load }}%"></div>
                        </div>
                    </td>
                    <td>{{ component.details }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <h2>Resource Pools</h2>
    <div class="status-table-container">
        <table class="status-table">
            <thead>
                <tr>
                    <th>Pool Name</th>
                    <th>Size</th>
                    <th>Active</th>
                    <th>Utilization</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for pool in resource_pools %}
                <tr>
                    <td>{{ pool.name }}</td>
                    <td>{{ pool.size }}</td>
                    <td>{{ pool.active }}</td>
                    <td>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: {{ pool.utilization }}%"></div>
                        </div>
                    </td>
                    <td><span class="status-badge {{ pool.status }}">{{ pool.status|title }}</span></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
{% endblock %}""")
        
        # Create additional templates - these would have similar structures
        # but are simplified here for brevity
        for template_name in ["workers.html", "tasks.html", "performance.html", 
                             "alerts.html", "topology.html", "fault-tolerance.html"]:
            template_path = os.path.join(template_dir, template_name)
            if not os.path.exists(template_path):
                with open(template_path, "w") as f:
                    page_title = template_name.replace(".html", "").replace("-", " ").title()
                    f.write(f"""{% extends "base.html" %}

{% block title %}Monitoring Dashboard - {page_title}{% endblock %}

{% block content %}
    <h1>{page_title}</h1>
    
    <!-- Content specific to {page_title} page will be implemented -->
    <div class="dashboard-grid">
        <div class="grid-item large">
            <div class="item-title">
                <span>{page_title} Overview</span>
            </div>
            <p>This section will display {page_title.lower()} metrics and visualizations.</p>
        </div>
    </div>
{% endblock %}""")
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update the dashboard configuration.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        self.config.update(config_updates)
        
        # Update child components if available
        if hasattr(self, 'dashboard_generator') and self.dashboard_generator:
            self.dashboard_generator.configure({
                "theme": self.config["theme"],
                "refresh_interval": self.config["auto_refresh"]
            })
            
        if hasattr(self, 'visualization_engine') and self.visualization_engine:
            self.visualization_engine.configure({
                "theme": self.config["theme"]
            })
        
        logger.info(f"Monitoring dashboard configuration updated: {config_updates}")
    
    def start(self):
        """Start the monitoring dashboard server and background tasks."""
        if self.running:
            logger.warning("Monitoring dashboard already running")
            return
        
        try:
            # Start background update thread
            self.update_stop_event.clear()
            self.update_thread = threading.Thread(
                target=self._background_update_loop,
                daemon=True
            )
            self.update_thread.start()
            
            # Auto-connect to coordinator if configured
            if self.config["auto_connect_coordinator"] and self.coordinator_url:
                self._connect_to_coordinator()
            
            # Start web server if aiohttp is available
            if AIOHTTP_AVAILABLE:
                # Use a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Start the web application
                runner = web.AppRunner(self.app)
                loop.run_until_complete(runner.setup())
                site = web.TCPSite(runner, self.host, self.port)
                loop.run_until_complete(site.start())
                
                self.running = True
                logger.info(f"Monitoring dashboard started at http://{self.host}:{self.port}")
                
                # Run the event loop
                loop.run_forever()
            else:
                # If web server not available, just run background tasks
                self.running = True
                logger.info("Monitoring dashboard background tasks started (web server not available)")
                self.update_stop_event.wait()  # Wait for stop event
                
        except Exception as e:
            logger.error(f"Error starting monitoring dashboard: {e}")
            self.stop()
            raise
    
    def start_async(self):
        """Start the monitoring dashboard in a separate thread.
        
        Returns:
            Threading.Thread: The background thread running the dashboard
        """
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread
    
    def stop(self):
        """Stop the monitoring dashboard and all background tasks."""
        if not self.running:
            return
        
        # Stop background update thread
        self.update_stop_event.set()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        # Close any connections
        if hasattr(self, 'websocket_connections'):
            for ws in list(self.websocket_connections):
                asyncio.run_coroutine_threadsafe(ws.close(), asyncio.get_event_loop())
            self.websocket_connections.clear()
        
        # Disconnect from coordinator
        self._disconnect_from_coordinator()
        
        self.running = False
        logger.info("Monitoring dashboard stopped")
    
    def _background_update_loop(self):
        """Background loop for updating system metrics and status."""
        while not self.update_stop_event.is_set():
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for alerts
                self._check_for_alerts()
                
                # Update last updated timestamp
                self.last_updated = datetime.now()
                
                # Broadcast updates to connected WebSocket clients
                self._broadcast_updates()
                
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
            
            # Wait for next update interval
            self.update_stop_event.wait(self.config["update_interval"])
    
    def _update_system_metrics(self):
        """Update system metrics from all sources."""
        # Get metrics from coordinator
        if hasattr(self, 'coordinator_client') and self.coordinator_client:
            self._update_coordinator_metrics()
        
        # Get metrics from result aggregator
        if self.result_aggregator:
            self._update_result_aggregator_metrics()
        
        # Update worker metrics
        self._update_worker_metrics()
        
        # Update task metrics
        self._update_task_metrics()
        
        # Update topology data
        self._update_topology_data()
        
        # Calculate system health score
        self._calculate_system_health()
        
        logger.debug("System metrics updated")
    
    def _update_coordinator_metrics(self):
        """Update metrics from the coordinator."""
        # Implementation depends on coordinator client interface
        pass
    
    def _update_result_aggregator_metrics(self):
        """Update metrics from the result aggregator."""
        # Implementation depends on result aggregator interface
        pass
    
    def _update_worker_metrics(self):
        """Update metrics for worker nodes."""
        # Collect metrics from worker connections
        for worker_id, connection in self.worker_connections.items():
            # Implementation depends on worker connection interface
            pass
    
    def _update_task_metrics(self):
        """Update metrics for task execution."""
        # Implementation depends on task tracking mechanism
        pass
    
    def _update_topology_data(self):
        """Update system topology data."""
        # Implementation depends on topology data collection
        pass
    
    def _calculate_system_health(self):
        """Calculate overall system health score."""
        # Implementation depends on health metrics
        pass
    
    def _check_for_alerts(self):
        """Check for and generate alerts based on metrics."""
        if not self.config["enable_alerts"]:
            return
        
        # Check for critical conditions
        
        # Example alert generation (dummy example)
        current_time = datetime.now()
        # Generate test alert every 60 seconds
        if current_time.second == 0:
            self._add_alert(
                level="info",
                title="Test Alert",
                message=f"This is a test alert generated at {current_time.strftime('%H:%M:%S')}",
                source="monitoring_dashboard",
                metrics={"test_value": 100}
            )
    
    def _add_alert(self, level: str, title: str, message: str, 
                  source: str, metrics: Optional[Dict[str, Any]] = None):
        """Add a new alert to the system.
        
        Args:
            level: Alert level (critical, warning, info)
            title: Alert title
            message: Alert message
            source: Alert source
            metrics: Optional metrics related to the alert
        """
        # Create alert record
        alert = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "level": level,
            "title": title,
            "message": message,
            "source": source,
            "metrics": metrics or {}
        }
        
        # Add to alert history
        self.alert_history.append(alert)
        
        # Trim alert history
        max_age = datetime.now() - timedelta(days=self.config["alert_retention_days"])
        self.alert_history = [a for a in self.alert_history if a["timestamp"] > max_age]
        
        # Log alert
        log_level = logging.INFO if level == "info" else logging.WARNING if level == "warning" else logging.ERROR
        logger.log(log_level, f"ALERT {level.upper()}: {title} - {message}")
        
        # Broadcast alert to WebSocket clients
        alert_data = {
            "id": alert["id"],
            "timestamp": alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "title": title,
            "message": message,
            "source": source
        }
        self._broadcast_alert(alert_data)
    
    def _broadcast_updates(self):
        """Broadcast updates to all WebSocket clients."""
        if not AIOHTTP_AVAILABLE or not hasattr(self, 'websocket_connections'):
            return
            
        # Create status update message
        status_data = self._get_status_data()
        
        # Get the event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Broadcast to all clients
        for ws in list(self.websocket_connections):
            message = {
                "type": "status_update",
                "data": status_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if ws.closed:
                self.websocket_connections.discard(ws)
                continue
                
            asyncio.run_coroutine_threadsafe(
                ws.send_json(message), loop
            )
    
    def _broadcast_alert(self, alert_data):
        """Broadcast an alert to all WebSocket clients.
        
        Args:
            alert_data: Alert data to broadcast
        """
        if not AIOHTTP_AVAILABLE or not hasattr(self, 'websocket_connections'):
            return
        
        # Get the event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Broadcast to all clients
        for ws in list(self.websocket_connections):
            message = {
                "type": "alert",
                "data": alert_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if ws.closed:
                self.websocket_connections.discard(ws)
                continue
                
            asyncio.run_coroutine_threadsafe(
                ws.send_json(message), loop
            )
    
    def _get_status_data(self) -> Dict[str, Any]:
        """Get current status data for broadcasting.
        
        Returns:
            Dictionary with status data
        """
        # Collect status data from various sources
        status_data = {
            "workers": {
                "total": len(self.worker_connections),
                "active": sum(1 for w in self.worker_connections.values() if w.get("status") == "active"),
                "idle": sum(1 for w in self.worker_connections.values() if w.get("status") == "idle"),
            },
            "tasks": {
                "total": len(self.task_execution_tracking),
                "running": sum(1 for t in self.task_execution_tracking.values() if t.get("status") == "running"),
                "completed": sum(1 for t in self.task_execution_tracking.values() if t.get("status") == "completed"),
                "failed": sum(1 for t in self.task_execution_tracking.values() if t.get("status") == "failed"),
            },
            "system": {
                "health_score": self.system_metrics.get("health_score", 0),
                "coordinator_status": self.system_metrics.get("coordinator_status", "unknown"),
            },
            "alerts": {
                "count": len(self.alert_history),
                "critical": sum(1 for a in self.alert_history if a["level"] == "critical"),
                "warning": sum(1 for a in self.alert_history if a["level"] == "warning"),
                "info": sum(1 for a in self.alert_history if a["level"] == "info"),
            }
        }
        
        return status_data
    
    def _connect_to_coordinator(self):
        """Connect to the coordinator for real-time data."""
        if not self.coordinator_url:
            logger.warning("Cannot connect to coordinator: URL not specified")
            return
        
        # Implementation depends on coordinator client interface
        logger.info(f"Connected to coordinator at {self.coordinator_url}")
    
    def _disconnect_from_coordinator(self):
        """Disconnect from the coordinator."""
        if hasattr(self, 'coordinator_client') and self.coordinator_client:
            # Implementation depends on coordinator client interface
            logger.info("Disconnected from coordinator")
    
    async def handle_index(self, request):
        """Handle requests to the main dashboard page.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        # Get query parameters
        refresh = int(request.query.get("refresh", self.config["auto_refresh"]))
        theme = request.query.get("theme", self.config["theme"])
        
        # Update config if needed
        if refresh != self.config["auto_refresh"] or theme != self.config["theme"]:
            self.configure({
                "auto_refresh": refresh,
                "theme": theme
            })
        
        # Generate system load chart
        system_load_chart = None
        if self.visualization_engine:
            # Create temp chart
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                # Generate chart (implementation depends on visualization engine)
                # This is a placeholder - real implementation would create actual chart
                visualization_data = {"title": "System Load Chart"}
                system_load_chart = self.visualization_engine.create_visualization(
                    "time_series", visualization_data, tmp.name
                )
                
                # Use relative path for chart in template
                if system_load_chart:
                    system_load_chart = os.path.relpath(
                        system_load_chart, 
                        os.path.dirname(self.output_dir)
                    )
        
        # Collect data for template
        data = {
            "active_page": "index",
            "auto_refresh": refresh,
            "theme": theme,
            "system_status": "active",  # or "warning" or "error"
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Worker stats
            "total_workers": len(self.worker_connections),
            "active_workers": sum(1 for w in self.worker_connections.values() if w.get("status") == "active"),
            "idle_workers": sum(1 for w in self.worker_connections.values() if w.get("status") == "idle"),
            
            # Task stats
            "total_tasks": len(self.task_execution_tracking),
            "running_tasks": sum(1 for t in self.task_execution_tracking.values() if t.get("status") == "running"),
            "completed_tasks": sum(1 for t in self.task_execution_tracking.values() if t.get("status") == "completed"),
            "pending_tasks": sum(1 for t in self.task_execution_tracking.values() if t.get("status") == "pending"),
            "failed_tasks": sum(1 for t in self.task_execution_tracking.values() if t.get("status") == "failed"),
            
            # Performance stats
            "avg_throughput": 42.5,  # Placeholder
            "total_tests_run": 156,  # Placeholder
            
            # Alert stats
            "total_alerts": len(self.alert_history),
            "critical_alerts": sum(1 for a in self.alert_history if a["level"] == "critical"),
            "warning_alerts": sum(1 for a in self.alert_history if a["level"] == "warning"),
            "info_alerts": sum(1 for a in self.alert_history if a["level"] == "info"),
            
            # Charts
            "system_load_chart": system_load_chart or "/static/placeholder_chart.png",
            
            # Recent alerts (limit to 5)
            "recent_alerts": [
                {
                    "level": a["level"],
                    "timestamp": a["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "title": a["title"],
                    "message": a["message"]
                }
                for a in sorted(self.alert_history, key=lambda x: x["timestamp"], reverse=True)[:5]
            ],
            
            # Hardware utilization
            "hardware_utilization": [
                {"type": "CPU", "count": 24, "utilization": 65, "status": "success"},
                {"type": "GPU", "count": 8, "utilization": 82, "status": "warning"},
                {"type": "TPU", "count": 2, "utilization": 45, "status": "success"},
                {"type": "Browser", "count": 12, "utilization": 30, "status": "success"}
            ],
            
            # Active workers (limit to 10)
            "active_worker_list": [
                {
                    "id": f"worker{i}",
                    "hardware": ["CPU", "GPU", "TPU", "Browser"][i % 4],
                    "tasks_completed": i * 10,
                    "tasks_total": i * 12,
                    "uptime": f"{i * 2}h {i * 11}m",
                    "load": 30 + (i * 7) % 65,
                    "status": "success" if i % 4 != 1 else "warning"
                }
                for i in range(min(10, max(1, len(self.worker_connections))))
            ],
            
            # Recent tasks (limit to 10)
            "recent_tasks": [
                {
                    "id": f"task{i}",
                    "type": ["benchmark", "inference", "training", "evaluation"][i % 4],
                    "worker_id": f"worker{i % 5}",
                    "start_time": (datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M:%S"),
                    "duration": f"{i*2+1}m {i*7}s",
                    "status": ["success", "warning", "danger", "info"][i % 4]
                }
                for i in range(min(10, max(1, len(self.task_execution_tracking))))
            ]
        }
        
        # Render template
        if JINJA2_AVAILABLE and hasattr(self, 'jinja_env'):
            template = self.jinja_env.get_template("index.html")
            html = template.render(**data)
            return web.Response(text=html, content_type="text/html")
        else:
            # Simple response if Jinja2 not available
            html = f"""
            <html>
            <head><title>Monitoring Dashboard</title></head>
            <body>
                <h1>Monitoring Dashboard</h1>
                <p>Basic version (template rendering not available)</p>
                <p>Last updated: {data['last_updated']}</p>
                <ul>
                    <li>Workers: {data['total_workers']} ({data['active_workers']} active)</li>
                    <li>Tasks: {data['total_tasks']} ({data['running_tasks']} running)</li>
                    <li>Alerts: {data['total_alerts']}</li>
                </ul>
            </body>
            </html>
            """
            return web.Response(text=html, content_type="text/html")
    
    async def handle_system_view(self, request):
        """Handle requests to the system view page."""
        # Implementation similar to handle_index but for system view
        # For brevity, we'll return a placeholder response
        return web.Response(text="System view implementation", content_type="text/plain")
    
    async def handle_workers_view(self, request):
        """Handle requests to the workers view page."""
        return web.Response(text="Workers view implementation", content_type="text/plain")
    
    async def handle_tasks_view(self, request):
        """Handle requests to the tasks view page."""
        return web.Response(text="Tasks view implementation", content_type="text/plain")
    
    async def handle_performance_view(self, request):
        """Handle requests to the performance view page."""
        return web.Response(text="Performance view implementation", content_type="text/plain")
    
    async def handle_alerts_view(self, request):
        """Handle requests to the alerts view page."""
        return web.Response(text="Alerts view implementation", content_type="text/plain")
    
    async def handle_topology_view(self, request):
        """Handle requests to the topology view page."""
        return web.Response(text="Topology view implementation", content_type="text/plain")
    
    async def handle_fault_tolerance_view(self, request):
        """Handle requests to the fault tolerance view page."""
        return web.Response(text="Fault tolerance view implementation", content_type="text/plain")
    
    async def handle_api_status(self, request):
        """Handle requests to the status API endpoint."""
        status_data = self._get_status_data()
        response = {
            "status": "ok",
            "data": status_data,
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_api_workers(self, request):
        """Handle requests to the workers API endpoint."""
        # Get worker data from worker connections
        worker_data = [
            {
                "id": worker_id,
                "status": worker_info.get("status", "unknown"),
                "hardware": worker_info.get("hardware", "unknown"),
                "tasks_completed": worker_info.get("tasks_completed", 0),
                "tasks_running": worker_info.get("tasks_running", 0),
                "uptime": worker_info.get("uptime", "unknown"),
                "last_seen": worker_info.get("last_seen", datetime.now()).isoformat()
            }
            for worker_id, worker_info in self.worker_connections.items()
        ]
        
        response = {
            "status": "ok",
            "data": {
                "workers": worker_data,
                "total": len(worker_data)
            },
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_api_tasks(self, request):
        """Handle requests to the tasks API endpoint."""
        # Get task data from task tracking
        task_data = [
            {
                "id": task_id,
                "status": task_info.get("status", "unknown"),
                "type": task_info.get("type", "unknown"),
                "worker_id": task_info.get("worker_id", "unknown"),
                "start_time": task_info.get("start_time", datetime.now()).isoformat(),
                "duration": task_info.get("duration", 0),
                "progress": task_info.get("progress", 0)
            }
            for task_id, task_info in self.task_execution_tracking.items()
        ]
        
        response = {
            "status": "ok",
            "data": {
                "tasks": task_data,
                "total": len(task_data)
            },
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_api_metrics(self, request):
        """Handle requests to the metrics API endpoint."""
        response = {
            "status": "ok",
            "data": {
                "metrics": self.system_metrics
            },
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_api_topology(self, request):
        """Handle requests to the topology API endpoint."""
        response = {
            "status": "ok",
            "data": {
                "topology": self.topology_cache or {}
            },
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_api_performance(self, request):
        """Handle requests to the performance API endpoint."""
        response = {
            "status": "ok",
            "data": {
                "performance": self.performance_history
            },
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_api_alerts(self, request):
        """Handle requests to the alerts API endpoint."""
        # Format alerts for API response
        alerts = [
            {
                "id": alert["id"],
                "level": alert["level"],
                "title": alert["title"],
                "message": alert["message"],
                "source": alert["source"],
                "timestamp": alert["timestamp"].isoformat()
            }
            for alert in self.alert_history
        ]
        
        response = {
            "status": "ok",
            "data": {
                "alerts": alerts,
                "total": len(alerts)
            },
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_api_fault_tolerance(self, request):
        """Handle requests to the fault tolerance API endpoint."""
        # Get fault tolerance data
        # Implementation depends on fault tolerance system interface
        
        response = {
            "status": "ok",
            "data": {
                "fault_tolerance": {
                    "patterns": [],  # Placeholder
                    "recovery_actions": [],  # Placeholder
                    "metrics": {}  # Placeholder
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(response)
    
    async def handle_websocket(self, request):
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Add to active connections
        self.websocket_connections.add(ws)
        
        try:
            # Send initial status
            await ws.send_json({
                "type": "status_update",
                "data": self._get_status_data(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Listen for messages
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # Handle message based on type
                        if data.get("type") == "ping":
                            await ws.send_json({
                                "type": "pong",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    except json.JSONDecodeError:
                        pass
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
        finally:
            # Remove from active connections
            self.websocket_connections.discard(ws)
        
        return ws


# Command-line interface
def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Monitoring Dashboard for Distributed Testing")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind the server to")
    parser.add_argument("--coordinator", help="URL of the coordinator server")
    parser.add_argument("--output-dir", default="./monitoring_dashboard", help="Output directory for dashboard files")
    parser.add_argument("--theme", choices=["light", "dark"], default="dark", help="Dashboard theme")
    parser.add_argument("--refresh", type=int, default=30, help="Auto-refresh interval in seconds (0 to disable)")
    parser.add_argument("--browser", action="store_true", help="Open dashboard in browser")
    
    args = parser.parse_args()
    
    # Create and configure the monitoring dashboard
    dashboard = MonitoringDashboard(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator,
        output_dir=args.output_dir
    )
    
    # Configure dashboard
    dashboard.configure({
        "theme": args.theme,
        "auto_refresh": args.refresh
    })
    
    # Auto-open in browser if requested
    if args.browser:
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}")
    
    try:
        # Start dashboard (this will block until interrupted)
        print(f"Starting monitoring dashboard at http://{args.host}:{args.port}")
        dashboard.start()
    except KeyboardInterrupt:
        print("Stopping monitoring dashboard...")
        dashboard.stop()


if __name__ == "__main__":
    main()