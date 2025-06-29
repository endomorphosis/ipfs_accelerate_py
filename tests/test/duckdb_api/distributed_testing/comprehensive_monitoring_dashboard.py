#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Monitoring Dashboard for Distributed Tests

This module implements a comprehensive monitoring dashboard for the Distributed Testing
Framework, integrating real-time worker status monitoring, task execution tracking,
error visualization, circuit breaker status, and performance analytics.

It provides a web-based interface to monitor the health and performance of the entire
distributed testing environment, with features for visualization, alerting, and interactive
exploration of test results and system metrics.
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import traceback
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path

import tornado.ioloop
import tornado.web
import tornado.websocket
import websockets
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports from other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import distributed testing components
from duckdb_api.distributed_testing.fault_tolerance_visualization import FaultToleranceVisualization
from duckdb_api.distributed_testing.monitoring_dashboard import DashboardMetrics
from duckdb_api.distributed_testing.result_aggregator.service import ResultAggregatorService
from duckdb_api.distributed_testing.circuit_breaker import CircuitBreakerRegistry
from duckdb_api.distributed_testing.dashboard.visualization import create_visualization


class ComprehensiveMonitoringDashboard:
    """
    Comprehensive monitoring dashboard for the Distributed Testing Framework.
    
    This class integrates various monitoring components into a unified dashboard:
    - Worker status monitoring
    - Task execution tracking
    - Error visualization
    - Circuit breaker status
    - Performance analytics
    - Resource utilization monitoring
    - Result aggregation visualization
    """
    
    def __init__(self, 
                 coordinator=None,
                 port=8888,
                 coordinator_url="http://localhost:8080",
                 db_path=None,
                 static_path=None,
                 template_path=None,
                 debug=False):
        """
        Initialize the comprehensive monitoring dashboard.
        
        Args:
            coordinator: The coordinator server instance (optional)
            port: Port for the dashboard web server
            coordinator_url: URL to connect to coordinator if not directly provided
            db_path: Path to the SQLite/DuckDB database
            static_path: Path to static files (CSS, JS, etc.)
            template_path: Path to HTML templates
            debug: Enable debug mode
        """
        self.coordinator = coordinator
        self.port = port
        self.coordinator_url = coordinator_url
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), "dashboard.db")
        
        # Set paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.static_path = static_path or os.path.join(self.base_dir, "static")
        self.template_path = template_path or os.path.join(self.base_dir, "templates")
        self.dashboard_path = os.path.join(self.base_dir, "dashboards")
        
        # Create directories if they don't exist
        os.makedirs(self.static_path, exist_ok=True)
        os.makedirs(self.template_path, exist_ok=True)
        os.makedirs(self.dashboard_path, exist_ok=True)
        
        # Dashboard components
        self.dashboard_metrics = DashboardMetrics(db_path=self.db_path)
        self.fault_tolerance_viz = None
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.result_aggregator = None
        
        # Dashboard web server
        self.app = None
        self.io_loop = None
        self.server = None
        self.debug = debug
        
        # WebSocket clients
        self.ws_clients = set()
        self.ws_clients_lock = threading.RLock()
        
        # Page registration
        self.registered_pages = {}
        
        # Visualization registration
        self.registered_visualizations = {}
        
        # Initialize visualization components
        if self.coordinator:
            self._initialize_visualization_components()
        
        logger.info(f"Comprehensive monitoring dashboard initialized on port {self.port}")
    
    def _initialize_visualization_components(self):
        """Initialize visualization components based on coordinator."""
        # Initialize fault tolerance visualization if fault tolerance system is available
        fault_tolerance_system = getattr(self.coordinator, 'fault_tolerance_system', None)
        if fault_tolerance_system:
            self.fault_tolerance_viz = FaultToleranceVisualization(
                fault_tolerance_system=fault_tolerance_system,
                dashboard_integration=self
            )
            logger.info("Fault tolerance visualization initialized")
        
        # Initialize result aggregator if result aggregator service is available
        result_aggregator_service = getattr(self.coordinator, 'result_aggregator_service', None)
        if result_aggregator_service:
            self.result_aggregator = result_aggregator_service
            logger.info("Result aggregator integration initialized")
        
        # Initialize circuit breaker registry if available
        circuit_breaker_registry = getattr(self.coordinator, 'circuit_breaker_registry', None)
        if circuit_breaker_registry:
            self.circuit_breaker_registry = circuit_breaker_registry
            logger.info("Circuit breaker registry integration initialized")
    
    def start(self):
        """Start the dashboard web server."""
        # Initialize Tornado application
        self.app = self._create_tornado_application()
        
        # Start the server
        self.server = self.app.listen(self.port)
        self.io_loop = tornado.ioloop.IOLoop.current()
        
        # Start periodic updates
        self._start_periodic_updates()
        
        logger.info(f"Dashboard server started on port {self.port}")
        
        # Start the IO loop (blocking call)
        try:
            self.io_loop.start()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the dashboard web server."""
        if self.server:
            self.server.stop()
            
        if self.io_loop:
            self.io_loop.add_callback(self.io_loop.stop)
            
        logger.info("Dashboard server stopped")
    
    def register_page(self, page_id, page_title, page_generator):
        """
        Register a page with the dashboard.
        
        Args:
            page_id: Unique identifier for the page
            page_title: Human-readable title for the page
            page_generator: Function that generates the page content
        """
        self.registered_pages[page_id] = {
            "title": page_title,
            "generator": page_generator
        }
        logger.info(f"Registered page: {page_id} - {page_title}")
    
    def register_visualization(self, vis_id, vis_title, vis_generator):
        """
        Register a visualization with the dashboard.
        
        Args:
            vis_id: Unique identifier for the visualization
            vis_title: Human-readable title for the visualization
            vis_generator: Function that generates the visualization
        """
        self.registered_visualizations[vis_id] = {
            "title": vis_title,
            "generator": vis_generator
        }
        logger.info(f"Registered visualization: {vis_id} - {vis_title}")
    
    def _create_tornado_application(self):
        """Create the Tornado web application with handlers."""
        handlers = [
            # Main dashboard
            (r"/", MainHandler, {"dashboard": self}),
            
            # System status
            (r"/status", StatusHandler, {"dashboard": self}),
            (r"/workers", WorkersHandler, {"dashboard": self}),
            (r"/tasks", TasksHandler, {"dashboard": self}),
            
            # Visualizations
            (r"/visualizations/([a-zA-Z0-9_-]+)", VisualizationHandler, {"dashboard": self}),
            
            # Pages
            (r"/pages/([a-zA-Z0-9_-]+)", PageHandler, {"dashboard": self}),
            
            # WebSocket
            (r"/ws", DashboardWebSocketHandler, {"dashboard": self}),
            
            # Static files
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.static_path}),
            
            # Dashboard files
            (r"/dashboards/(.*)", tornado.web.StaticFileHandler, {"path": self.dashboard_path})
        ]
        
        settings = {
            "template_path": self.template_path,
            "static_path": self.static_path,
            "debug": self.debug
        }
        
        return tornado.web.Application(handlers, **settings)
    
    def _start_periodic_updates(self):
        """Start periodic update callbacks."""
        # Update dashboard metrics every 10 seconds
        self.io_loop.add_callback(self._update_metrics)
    
    async def _update_metrics(self):
        """Periodically update dashboard metrics and notify clients."""
        while True:
            try:
                # Get updated metrics
                metrics = self._collect_system_metrics()
                
                # Broadcast metrics to all connected clients
                self._broadcast_update("metrics", metrics)
                
                # Wait for next update cycle
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.exception(f"Error updating metrics: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "workers": self._get_worker_metrics(),
            "tasks": self._get_task_metrics(),
            "resources": self._get_resource_metrics(),
            "error_rate": self._get_error_rate_metrics()
        }
        
        return metrics
    
    def _get_worker_metrics(self):
        """Get worker metrics."""
        # If coordinator is available, get real metrics
        if self.coordinator and hasattr(self.coordinator, 'worker_manager'):
            worker_manager = self.coordinator.worker_manager
            workers = worker_manager.workers
            
            worker_metrics = {
                "total": len(workers),
                "active": sum(1 for w in workers.values() if w.get("status") == "active"),
                "inactive": sum(1 for w in workers.values() if w.get("status") == "inactive"),
                "disconnected": sum(1 for w in workers.values() if w.get("status") == "disconnected"),
                "by_hardware": defaultdict(int)
            }
            
            # Count workers by hardware type
            for worker in workers.values():
                if "hardware" in worker:
                    for hw_type in worker["hardware"]:
                        worker_metrics["by_hardware"][hw_type] += 1
            
            return worker_metrics
        
        # Otherwise, return simulated metrics
        return {
            "total": 10,
            "active": 8,
            "inactive": 1,
            "disconnected": 1,
            "by_hardware": {
                "cpu": 10,
                "cuda": 6,
                "rocm": 2,
                "webgpu": 3
            }
        }
    
    def _get_task_metrics(self):
        """Get task metrics."""
        # If coordinator is available, get real metrics
        if self.coordinator and hasattr(self.coordinator, 'task_manager'):
            task_manager = self.coordinator.task_manager
            tasks = task_manager.tasks
            
            task_metrics = {
                "total": len(tasks),
                "pending": sum(1 for t in tasks.values() if t.get("status") == "pending"),
                "running": sum(1 for t in tasks.values() if t.get("status") == "running"),
                "completed": sum(1 for t in tasks.values() if t.get("status") == "completed"),
                "failed": sum(1 for t in tasks.values() if t.get("status") == "failed"),
                "cancelled": sum(1 for t in tasks.values() if t.get("status") == "cancelled"),
                "by_type": defaultdict(int)
            }
            
            # Count tasks by type
            for task in tasks.values():
                task_type = task.get("type", "unknown")
                task_metrics["by_type"][task_type] += 1
            
            return task_metrics
        
        # Otherwise, return simulated metrics
        return {
            "total": 25,
            "pending": 5,
            "running": 8,
            "completed": 10,
            "failed": 1,
            "cancelled": 1,
            "by_type": {
                "benchmark": 15,
                "test": 8,
                "command": 2
            }
        }
    
    def _get_resource_metrics(self):
        """Get resource utilization metrics."""
        # If coordinator is available, get real metrics
        if self.coordinator and hasattr(self.coordinator, 'resource_manager'):
            resource_manager = self.coordinator.resource_manager
            
            # In a real implementation, we would get actual resource metrics
            # For now, just return a structure with placeholder values
            return {
                "cpu_utilization": 0.65,
                "memory_utilization": 0.48,
                "gpu_utilization": 0.72,
                "network_bandwidth": 0.35
            }
        
        # Otherwise, return simulated metrics
        return {
            "cpu_utilization": 0.65,
            "memory_utilization": 0.48,
            "gpu_utilization": 0.72,
            "network_bandwidth": 0.35
        }
    
    def _get_error_rate_metrics(self):
        """Get error rate metrics."""
        # If coordinator is available, get real metrics
        if self.coordinator and hasattr(self.coordinator, 'fault_tolerance_system'):
            fault_tolerance_system = self.coordinator.fault_tolerance_system
            
            # Get error rate from fault tolerance system
            return {
                "current": len(fault_tolerance_system.error_history) / fault_tolerance_system.error_window_size if fault_tolerance_system.error_history else 0,
                "threshold": fault_tolerance_system.error_rate_threshold,
                "severity_distribution": {
                    "low": sum(1 for e in fault_tolerance_system.error_history if e["severity"].value == "low"),
                    "medium": sum(1 for e in fault_tolerance_system.error_history if e["severity"].value == "medium"),
                    "high": sum(1 for e in fault_tolerance_system.error_history if e["severity"].value == "high"),
                    "critical": sum(1 for e in fault_tolerance_system.error_history if e["severity"].value == "critical")
                }
            }
        
        # Otherwise, return simulated metrics
        return {
            "current": 0.05,
            "threshold": 0.5,
            "severity_distribution": {
                "low": 3,
                "medium": 2,
                "high": 0,
                "critical": 0
            }
        }
    
    def _broadcast_update(self, update_type, data):
        """Broadcast an update to all connected WebSocket clients."""
        message = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        with self.ws_clients_lock:
            for client in self.ws_clients:
                try:
                    client.write_message(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending to WebSocket client: {e}")
    
    def _register_client(self, client):
        """Register a new WebSocket client."""
        with self.ws_clients_lock:
            self.ws_clients.add(client)
            logger.info(f"Client connected, total clients: {len(self.ws_clients)}")
    
    def _unregister_client(self, client):
        """Unregister a WebSocket client."""
        with self.ws_clients_lock:
            if client in self.ws_clients:
                self.ws_clients.remove(client)
                logger.info(f"Client disconnected, total clients: {len(self.ws_clients)}")
    
    def generate_system_overview(self):
        """Generate a system overview visualization."""
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Worker Status",
                "Task Status",
                "Resource Utilization",
                "Error Rate"
            ),
            specs=[
                [{"type": "pie"}, {"type": "pie"}],
                [{"type": "indicator"}, {"type": "indicator"}]
            ]
        )
        
        # Get current metrics
        metrics = self._collect_system_metrics()
        
        # Worker Status Pie Chart
        worker_metrics = metrics["workers"]
        fig.add_trace(
            go.Pie(
                labels=["Active", "Inactive", "Disconnected"],
                values=[worker_metrics["active"], worker_metrics["inactive"], worker_metrics["disconnected"]],
                marker=dict(colors=["#4CAF50", "#FFC107", "#F44336"]),
                textinfo="percent+label",
                hole=0.3
            ),
            row=1, col=1
        )
        
        # Task Status Pie Chart
        task_metrics = metrics["tasks"]
        fig.add_trace(
            go.Pie(
                labels=["Pending", "Running", "Completed", "Failed", "Cancelled"],
                values=[task_metrics["pending"], task_metrics["running"], task_metrics["completed"], task_metrics["failed"], task_metrics["cancelled"]],
                marker=dict(colors=["#2196F3", "#4CAF50", "#9C27B0", "#F44336", "#FF9800"]),
                textinfo="percent+label",
                hole=0.3
            ),
            row=1, col=2
        )
        
        # Resource Utilization Gauge
        resource_metrics = metrics["resources"]
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=resource_metrics["cpu_utilization"] * 100,
                title={"text": "CPU Utilization (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2196F3"},
                    "steps": [
                        {"range": [0, 50], "color": "#E0F7FA"},
                        {"range": [50, 75], "color": "#B2EBF2"},
                        {"range": [75, 90], "color": "#80DEEA"},
                        {"range": [90, 100], "color": "#4DD0E1"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ),
            row=2, col=1
        )
        
        # Error Rate Gauge
        error_metrics = metrics["error_rate"]
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=error_metrics["current"] * 100,
                title={"text": "Error Rate (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#F44336"},
                    "steps": [
                        {"range": [0, 10], "color": "#E0F2F1"},
                        {"range": [10, 30], "color": "#B2DFDB"},
                        {"range": [30, 50], "color": "#80CBC4"},
                        {"range": [50, 100], "color": "#4DB6AC"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": error_metrics["threshold"] * 100
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="System Overview",
            height=800,
            width=1200,
            showlegend=True
        )
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_overview_{timestamp}.html"
        filepath = os.path.join(self.dashboard_path, filename)
        
        try:
            fig.write_html(filepath)
            logger.info(f"System overview visualization saved to {filepath}")
        except Exception as e:
            logger.exception(f"Error saving visualization: {e}")
        
        return {
            "figure": fig,
            "html_path": filepath,
            "metrics": metrics
        }
    
    def generate_worker_performance_visualization(self):
        """Generate a worker performance visualization."""
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Worker Task Throughput",
                "Worker Error Rates",
                "Task Execution Time by Worker",
                "Worker Hardware Distribution"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "pie"}]
            ]
        )
        
        # Simulated data for worker performance
        # In a real implementation, this would be fetched from the database
        worker_ids = [f"worker-{i}" for i in range(1, 6)]
        
        # Worker Task Throughput
        task_throughput = [random.randint(5, 20) for _ in range(5)]
        fig.add_trace(
            go.Bar(
                x=worker_ids,
                y=task_throughput,
                marker=dict(color="#1E88E5"),
                name="Tasks Completed"
            ),
            row=1, col=1
        )
        
        # Worker Error Rates
        error_rates = [random.uniform(0, 0.15) for _ in range(5)]
        fig.add_trace(
            go.Bar(
                x=worker_ids,
                y=[rate * 100 for rate in error_rates],  # Convert to percentage
                marker=dict(color="#F44336"),
                name="Error Rate (%)"
            ),
            row=1, col=2
        )
        
        # Add threshold line
        fig.add_trace(
            go.Scatter(
                x=[worker_ids[0], worker_ids[-1]],
                y=[10, 10],  # 10% threshold
                mode="lines",
                name="Error Threshold",
                line=dict(color="#F44336", width=2, dash="dash")
            ),
            row=1, col=2
        )
        
        # Task Execution Time by Worker
        # Simulated execution times for each worker
        execution_times = [
            [random.uniform(0.5, 2.0) for _ in range(10)] for _ in range(5)
        ]
        
        for i, worker_id in enumerate(worker_ids):
            fig.add_trace(
                go.Box(
                    y=execution_times[i],
                    name=worker_id,
                    boxmean=True
                ),
                row=2, col=1
            )
        
        # Worker Hardware Distribution
        hardware_types = ["CPU", "CUDA", "ROCm", "WebGPU", "WebNN"]
        hardware_counts = [5, 3, 1, 2, 1]  # Simulated counts
        
        fig.add_trace(
            go.Pie(
                labels=hardware_types,
                values=hardware_counts,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo="percent+label",
                hole=0.3
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Worker Performance Analysis",
            height=800,
            width=1200,
            showlegend=True
        )
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"worker_performance_{timestamp}.html"
        filepath = os.path.join(self.dashboard_path, filename)
        
        try:
            fig.write_html(filepath)
            logger.info(f"Worker performance visualization saved to {filepath}")
        except Exception as e:
            logger.exception(f"Error saving visualization: {e}")
        
        return {
            "figure": fig,
            "html_path": filepath
        }
    
    def generate_task_performance_visualization(self):
        """Generate a task performance visualization."""
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Task Execution Time by Type",
                "Task Status Distribution",
                "Task Completion Rate Over Time",
                "Task Type Distribution"
            ),
            specs=[
                [{"type": "box"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "pie"}]
            ]
        )
        
        # Simulated data for task performance
        # In a real implementation, this would be fetched from the database
        task_types = ["benchmark", "test", "command"]
        
        # Task Execution Time by Type
        execution_times = {
            "benchmark": [random.uniform(10, 30) for _ in range(15)],
            "test": [random.uniform(5, 15) for _ in range(8)],
            "command": [random.uniform(1, 5) for _ in range(2)]
        }
        
        for task_type in task_types:
            fig.add_trace(
                go.Box(
                    y=execution_times[task_type],
                    name=task_type,
                    boxmean=True
                ),
                row=1, col=1
            )
        
        # Task Status Distribution
        task_metrics = self._get_task_metrics()
        fig.add_trace(
            go.Pie(
                labels=["Pending", "Running", "Completed", "Failed", "Cancelled"],
                values=[task_metrics["pending"], task_metrics["running"], task_metrics["completed"], task_metrics["failed"], task_metrics["cancelled"]],
                marker=dict(colors=["#2196F3", "#4CAF50", "#9C27B0", "#F44336", "#FF9800"]),
                textinfo="percent+label",
                hole=0.3
            ),
            row=1, col=2
        )
        
        # Task Completion Rate Over Time
        # Simulated time series data
        timestamps = [datetime.now() - timedelta(hours=x) for x in range(24, 0, -1)]
        completion_rates = [random.uniform(0.8, 1.0) for _ in range(24)]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[rate * 100 for rate in completion_rates],  # Convert to percentage
                mode="lines+markers",
                name="Completion Rate (%)",
                line=dict(color="#4CAF50", width=2),
                marker=dict(size=6, color="#4CAF50")
            ),
            row=2, col=1
        )
        
        # Add target line
        fig.add_trace(
            go.Scatter(
                x=[timestamps[0], timestamps[-1]],
                y=[95, 95],  # 95% target
                mode="lines",
                name="Target Rate",
                line=dict(color="#FF9800", width=2, dash="dash")
            ),
            row=2, col=1
        )
        
        # Task Type Distribution
        fig.add_trace(
            go.Pie(
                labels=list(task_metrics["by_type"].keys()),
                values=list(task_metrics["by_type"].values()),
                marker=dict(colors=px.colors.qualitative.Pastel),
                textinfo="percent+label",
                hole=0.3
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Task Performance Analysis",
            height=800,
            width=1200,
            showlegend=True
        )
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task_performance_{timestamp}.html"
        filepath = os.path.join(self.dashboard_path, filename)
        
        try:
            fig.write_html(filepath)
            logger.info(f"Task performance visualization saved to {filepath}")
        except Exception as e:
            logger.exception(f"Error saving visualization: {e}")
        
        return {
            "figure": fig,
            "html_path": filepath
        }
    
    def generate_resource_utilization_visualization(self):
        """Generate a resource utilization visualization."""
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "CPU Utilization Over Time",
                "Memory Utilization Over Time",
                "GPU Utilization Over Time",
                "Resource Utilization by Worker"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Simulated data for resource utilization
        # In a real implementation, this would be fetched from the database
        timestamps = [datetime.now() - timedelta(minutes=x) for x in range(60, 0, -1)]
        
        # CPU Utilization
        cpu_values = [random.uniform(0.3, 0.8) for _ in range(60)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[val * 100 for val in cpu_values],  # Convert to percentage
                mode="lines",
                name="CPU Utilization (%)",
                line=dict(color="#2196F3", width=2)
            ),
            row=1, col=1
        )
        
        # Memory Utilization
        memory_values = [random.uniform(0.4, 0.6) for _ in range(60)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[val * 100 for val in memory_values],  # Convert to percentage
                mode="lines",
                name="Memory Utilization (%)",
                line=dict(color="#9C27B0", width=2)
            ),
            row=1, col=2
        )
        
        # GPU Utilization
        gpu_values = [random.uniform(0.5, 0.9) for _ in range(60)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[val * 100 for val in gpu_values],  # Convert to percentage
                mode="lines",
                name="GPU Utilization (%)",
                line=dict(color="#FF9800", width=2)
            ),
            row=2, col=1
        )
        
        # Resource Utilization by Worker
        worker_ids = [f"worker-{i}" for i in range(1, 6)]
        cpu_by_worker = [random.uniform(0.3, 0.8) for _ in range(5)]
        memory_by_worker = [random.uniform(0.4, 0.6) for _ in range(5)]
        gpu_by_worker = [random.uniform(0.5, 0.9) for _ in range(5)]
        
        fig.add_trace(
            go.Bar(
                x=worker_ids,
                y=[val * 100 for val in cpu_by_worker],
                name="CPU (%)",
                marker=dict(color="#2196F3")
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=worker_ids,
                y=[val * 100 for val in memory_by_worker],
                name="Memory (%)",
                marker=dict(color="#9C27B0")
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=worker_ids,
                y=[val * 100 for val in gpu_by_worker],
                name="GPU (%)",
                marker=dict(color="#FF9800")
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Resource Utilization Analysis",
            height=800,
            width=1200,
            showlegend=True,
            barmode="group"
        )
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resource_utilization_{timestamp}.html"
        filepath = os.path.join(self.dashboard_path, filename)
        
        try:
            fig.write_html(filepath)
            logger.info(f"Resource utilization visualization saved to {filepath}")
        except Exception as e:
            logger.exception(f"Error saving visualization: {e}")
        
        return {
            "figure": fig,
            "html_path": filepath
        }
    
    def generate_dashboard_index(self):
        """Generate the dashboard index HTML."""
        # Get registered pages
        pages = [(page_id, info["title"]) for page_id, info in self.registered_pages.items()]
        
        # Get registered visualizations
        visualizations = [(vis_id, info["title"]) for vis_id, info in self.registered_visualizations.items()]
        
        # Default visualizations if none registered
        if not visualizations:
            visualizations = [
                ("system_overview", "System Overview"),
                ("worker_performance", "Worker Performance"),
                ("task_performance", "Task Performance"),
                ("resource_utilization", "Resource Utilization")
            ]
        
        # Default pages if none registered
        if not pages:
            pages = [
                ("status", "System Status"),
                ("workers", "Worker Status"),
                ("tasks", "Task Status")
            ]
            
            # Add fault tolerance page if available
            if self.fault_tolerance_viz:
                pages.append(("fault_tolerance", "Fault Tolerance"))
            
            # Add result aggregation page if available
            if self.result_aggregator:
                pages.append(("result_aggregation", "Result Aggregation"))
        
        # Generate index HTML
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Monitoring Dashboard</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="/static/js/dashboard.js"></script>
</head>
<body>
    <div class="header">
        <h1>Comprehensive Monitoring Dashboard</h1>
        <p>Generated on {timestamp}</p>
    </div>
    
    <div class="dashboard-grid">
        <div class="sidebar">
            <h2>Navigation</h2>
            <ul class="nav-list">
                <li><a href="/" class="active">Dashboard Home</a></li>
                
                <li class="nav-header">System Status</li>
                {
                    chr(10).join([f'<li><a href="/pages/{page_id}">{title}</a></li>' for page_id, title in pages])
                }
                
                <li class="nav-header">Visualizations</li>
                {
                    chr(10).join([f'<li><a href="/visualizations/{vis_id}">{title}</a></li>' for vis_id, title in visualizations])
                }
            </ul>
        </div>
        
        <div class="main-content">
            <div class="dashboard-section">
                <h2>System Overview</h2>
                <iframe src="/visualizations/system_overview" class="dashboard-iframe"></iframe>
            </div>
            
            <div class="dashboard-row">
                <div class="dashboard-section half-width">
                    <h2>Worker Performance</h2>
                    <iframe src="/visualizations/worker_performance" class="dashboard-iframe"></iframe>
                </div>
                
                <div class="dashboard-section half-width">
                    <h2>Task Performance</h2>
                    <iframe src="/visualizations/task_performance" class="dashboard-iframe"></iframe>
                </div>
            </div>
            
            <div class="dashboard-section">
                <h2>Resource Utilization</h2>
                <iframe src="/visualizations/resource_utilization" class="dashboard-iframe"></iframe>
            </div>
            
            {
                '<div class="dashboard-section"><h2>Fault Tolerance</h2><iframe src="/pages/fault_tolerance" class="dashboard-iframe"></iframe></div>' 
                if self.fault_tolerance_viz else ''
            }
            
            {
                '<div class="dashboard-section"><h2>Result Aggregation</h2><iframe src="/pages/result_aggregation" class="dashboard-iframe"></iframe></div>' 
                if self.result_aggregator else ''
            }
        </div>
    </div>
    
    <div class="footer">
        <p>Distributed Testing Framework - Comprehensive Monitoring Dashboard</p>
    </div>
</body>
</html>
"""
        
        # Save index HTML
        index_path = os.path.join(self.template_path, "index.html")
        
        try:
            with open(index_path, "w") as f:
                f.write(index_html)
            logger.info(f"Dashboard index saved to {index_path}")
        except Exception as e:
            logger.exception(f"Error saving dashboard index: {e}")
        
        return index_html


# Tornado request handlers

class MainHandler(tornado.web.RequestHandler):
    """Handler for the main dashboard page."""
    
    def initialize(self, dashboard):
        """Initialize with dashboard reference."""
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for main page."""
        # Generate dashboard index
        index_html = self.dashboard.generate_dashboard_index()
        
        # Render index template
        self.write(index_html)


class StatusHandler(tornado.web.RequestHandler):
    """Handler for system status page."""
    
    def initialize(self, dashboard):
        """Initialize with dashboard reference."""
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for status page."""
        # Generate system overview
        overview = self.dashboard.generate_system_overview()
        
        # Render status template
        self.render("status.html", 
                    title="System Status",
                    overview_html=f"/dashboards/{os.path.basename(overview['html_path'])}",
                    metrics=overview["metrics"])


class WorkersHandler(tornado.web.RequestHandler):
    """Handler for worker status page."""
    
    def initialize(self, dashboard):
        """Initialize with dashboard reference."""
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for workers page."""
        # Generate worker performance visualization
        performance = self.dashboard.generate_worker_performance_visualization()
        
        # Render workers template
        self.render("workers.html", 
                    title="Worker Status",
                    performance_html=f"/dashboards/{os.path.basename(performance['html_path'])}")


class TasksHandler(tornado.web.RequestHandler):
    """Handler for task status page."""
    
    def initialize(self, dashboard):
        """Initialize with dashboard reference."""
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for tasks page."""
        # Generate task performance visualization
        performance = self.dashboard.generate_task_performance_visualization()
        
        # Render tasks template
        self.render("tasks.html", 
                    title="Task Status",
                    performance_html=f"/dashboards/{os.path.basename(performance['html_path'])}")


class VisualizationHandler(tornado.web.RequestHandler):
    """Handler for visualization pages."""
    
    def initialize(self, dashboard):
        """Initialize with dashboard reference."""
        self.dashboard = dashboard
    
    def get(self, vis_id):
        """Handle GET request for visualization page."""
        # Check if visualization is registered
        if vis_id in self.dashboard.registered_visualizations:
            # Generate visualization using registered generator
            vis_info = self.dashboard.registered_visualizations[vis_id]
            vis_data = vis_info["generator"]()
            
            # Render visualization template
            self.render("visualization.html", 
                        title=vis_info["title"],
                        vis_id=vis_id,
                        vis_html=f"/dashboards/{os.path.basename(vis_data['html_path'])}")
            return
        
        # Handle default visualizations
        if vis_id == "system_overview":
            overview = self.dashboard.generate_system_overview()
            self.redirect(f"/dashboards/{os.path.basename(overview['html_path'])}")
            
        elif vis_id == "worker_performance":
            performance = self.dashboard.generate_worker_performance_visualization()
            self.redirect(f"/dashboards/{os.path.basename(performance['html_path'])}")
            
        elif vis_id == "task_performance":
            performance = self.dashboard.generate_task_performance_visualization()
            self.redirect(f"/dashboards/{os.path.basename(performance['html_path'])}")
            
        elif vis_id == "resource_utilization":
            utilization = self.dashboard.generate_resource_utilization_visualization()
            self.redirect(f"/dashboards/{os.path.basename(utilization['html_path'])}")
            
        else:
            # Visualization not found
            self.set_status(404)
            self.write(f"Visualization '{vis_id}' not found")


class PageHandler(tornado.web.RequestHandler):
    """Handler for custom dashboard pages."""
    
    def initialize(self, dashboard):
        """Initialize with dashboard reference."""
        self.dashboard = dashboard
    
    def get(self, page_id):
        """Handle GET request for custom page."""
        # Check if page is registered
        if page_id in self.dashboard.registered_pages:
            # Generate page using registered generator
            page_info = self.dashboard.registered_pages[page_id]
            page_data = page_info["generator"]()
            
            # Render page template
            self.render("page.html", 
                        title=page_info["title"],
                        page_id=page_id,
                        page_data=page_data)
        else:
            # Page not found
            self.set_status(404)
            self.write(f"Page '{page_id}' not found")


class DashboardWebSocketHandler(tornado.websocket.WebSocketHandler):
    """WebSocket handler for real-time dashboard updates."""
    
    def initialize(self, dashboard):
        """Initialize with dashboard reference."""
        self.dashboard = dashboard
    
    def open(self):
        """Handle WebSocket connection opened."""
        self.dashboard._register_client(self)
    
    def on_message(self, message):
        """Handle WebSocket message received."""
        try:
            # Parse message as JSON
            data = json.loads(message)
            
            # Handle message based on type
            message_type = data.get("type")
            
            if message_type == "request_metrics":
                # Send current metrics
                metrics = self.dashboard._collect_system_metrics()
                self.write_message(json.dumps({
                    "type": "metrics",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat()
                }))
                
            elif message_type == "request_visualization":
                # Generate and send visualization
                vis_id = data.get("vis_id")
                if vis_id and vis_id in self.dashboard.registered_visualizations:
                    vis_info = self.dashboard.registered_visualizations[vis_id]
                    vis_data = vis_info["generator"]()
                    self.write_message(json.dumps({
                        "type": "visualization",
                        "vis_id": vis_id,
                        "vis_html": f"/dashboards/{os.path.basename(vis_data['html_path'])}",
                        "timestamp": datetime.now().isoformat()
                    }))
            
        except Exception as e:
            logger.exception(f"Error handling WebSocket message: {e}")
    
    def on_close(self):
        """Handle WebSocket connection closed."""
        self.dashboard._unregister_client(self)


def create_css_file(output_dir):
    """Create CSS file for the dashboard."""
    css_dir = os.path.join(output_dir, "css")
    os.makedirs(css_dir, exist_ok=True)
    
    css_path = os.path.join(css_dir, "dashboard.css")
    
    with open(css_path, "w") as f:
        f.write("""
/* Dashboard styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
    color: #333;
}

.header {
    background-color: #3f51b5;
    color: white;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

.header h1 {
    margin: 0;
    font-weight: 300;
}

.dashboard-grid {
    display: flex;
    min-height: calc(100vh - 140px);
}

.sidebar {
    width: 250px;
    background-color: #fff;
    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    padding: 20px;
}

.main-content {
    flex: 1;
    padding: 20px;
}

.dashboard-section {
    background-color: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 4px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.dashboard-row {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}

.half-width {
    width: calc(50% - 10px);
}

.dashboard-iframe {
    width: 100%;
    height: 600px;
    border: none;
}

.nav-list {
    list-style-type: none;
    padding: 0;
    margin: 0;
}

.nav-list li {
    margin-bottom: 10px;
}

.nav-list li a {
    display: block;
    padding: 10px;
    text-decoration: none;
    color: #333;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.nav-list li a:hover {
    background-color: #f0f0f0;
}

.nav-list li a.active {
    background-color: #e0e0e0;
    font-weight: bold;
}

.nav-header {
    font-weight: bold;
    margin-top: 20px;
    padding: 5px 10px;
    color: #757575;
    border-bottom: 1px solid #eee;
}

.footer {
    background-color: #3f51b5;
    color: white;
    text-align: center;
    padding: 10px;
    margin-top: 20px;
}

/* Responsive design */
@media (max-width: 768px) {
    .dashboard-grid {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
    }
    
    .dashboard-row {
        flex-direction: column;
    }
    
    .half-width {
        width: 100%;
    }
}
""")
    
    logger.info(f"CSS file created at {css_path}")


def create_js_file(output_dir):
    """Create JavaScript file for the dashboard."""
    js_dir = os.path.join(output_dir, "js")
    os.makedirs(js_dir, exist_ok=True)
    
    js_path = os.path.join(js_dir, "dashboard.js")
    
    with open(js_path, "w") as f:
        f.write("""
// Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Connect to WebSocket for real-time updates
    connectWebSocket();
    
    // Initialize dashboard
    initializeDashboard();
});

// WebSocket connection
let socket = null;

function connectWebSocket() {
    // Get the current host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;
    
    // Create WebSocket connection
    socket = new WebSocket(wsUrl);
    
    // Connection opened
    socket.addEventListener('open', function(event) {
        console.log('Connected to dashboard WebSocket');
        
        // Request initial metrics
        socket.send(JSON.stringify({
            type: 'request_metrics'
        }));
    });
    
    // Listen for messages
    socket.addEventListener('message', function(event) {
        const message = JSON.parse(event.data);
        
        // Handle message based on type
        if (message.type === 'metrics') {
            updateMetrics(message.data);
        } else if (message.type === 'visualization') {
            updateVisualization(message.vis_id, message.vis_html);
        }
    });
    
    // Connection closed
    socket.addEventListener('close', function(event) {
        console.log('Disconnected from dashboard WebSocket');
        
        // Reconnect after delay
        setTimeout(connectWebSocket, 2000);
    });
    
    // Connection error
    socket.addEventListener('error', function(event) {
        console.error('WebSocket error:', event);
    });
}

function initializeDashboard() {
    // Set active navigation item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-list li a');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
    
    // Set up periodic refresh
    setInterval(function() {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'request_metrics'
            }));
        }
    }, 10000); // Refresh every 10 seconds
}

function updateMetrics(metrics) {
    // Update metrics displays if they exist
    if (document.getElementById('worker-count')) {
        document.getElementById('worker-count').textContent = metrics.workers.total;
    }
    
    if (document.getElementById('active-workers')) {
        document.getElementById('active-workers').textContent = metrics.workers.active;
    }
    
    if (document.getElementById('task-count')) {
        document.getElementById('task-count').textContent = metrics.tasks.total;
    }
    
    if (document.getElementById('running-tasks')) {
        document.getElementById('running-tasks').textContent = metrics.tasks.running;
    }
    
    if (document.getElementById('error-rate')) {
        document.getElementById('error-rate').textContent = 
            (metrics.error_rate.current * 100).toFixed(2) + '%';
    }
    
    // Trigger custom event for metrics update
    const event = new CustomEvent('metricsUpdated', { detail: metrics });
    document.dispatchEvent(event);
}

function updateVisualization(visId, visHtml) {
    // Update visualization iframe if it exists
    const iframe = document.querySelector(`[data-vis-id="${visId}"]`);
    if (iframe) {
        iframe.src = visHtml;
    }
}

function requestVisualization(visId) {
    // Request visualization update
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'request_visualization',
            vis_id: visId
        }));
    }
}
""")
    
    logger.info(f"JavaScript file created at {js_path}")


def create_template_files(output_dir):
    """Create template files for the dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create status template
    status_path = os.path.join(output_dir, "status.html")
    with open(status_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Comprehensive Monitoring Dashboard</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="/static/js/dashboard.js"></script>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
    </div>
    
    <div class="dashboard-grid">
        <div class="sidebar">
            <h2>Navigation</h2>
            <ul class="nav-list">
                <li><a href="/">Dashboard Home</a></li>
                
                <li class="nav-header">System Status</li>
                <li><a href="/pages/status" class="active">System Status</a></li>
                <li><a href="/pages/workers">Worker Status</a></li>
                <li><a href="/pages/tasks">Task Status</a></li>
                
                <li class="nav-header">Visualizations</li>
                <li><a href="/visualizations/system_overview">System Overview</a></li>
                <li><a href="/visualizations/worker_performance">Worker Performance</a></li>
                <li><a href="/visualizations/task_performance">Task Performance</a></li>
                <li><a href="/visualizations/resource_utilization">Resource Utilization</a></li>
            </ul>
        </div>
        
        <div class="main-content">
            <div class="dashboard-section">
                <h2>System Overview</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Workers</h3>
                        <div class="metric-value" id="worker-count">{{ metrics.workers.total }}</div>
                        <div class="metric-detail">Active: <span id="active-workers">{{ metrics.workers.active }}</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Tasks</h3>
                        <div class="metric-value" id="task-count">{{ metrics.tasks.total }}</div>
                        <div class="metric-detail">Running: <span id="running-tasks">{{ metrics.tasks.running }}</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Error Rate</h3>
                        <div class="metric-value" id="error-rate">{{ "%.2f"|format(metrics.error_rate.current * 100) }}%</div>
                        <div class="metric-detail">Threshold: {{ "%.2f"|format(metrics.error_rate.threshold * 100) }}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Resources</h3>
                        <div class="metric-value">{{ "%.2f"|format(metrics.resources.cpu_utilization * 100) }}%</div>
                        <div class="metric-detail">CPU Utilization</div>
                    </div>
                </div>
            </div>
            
            <div class="dashboard-section">
                <h2>System Visualization</h2>
                <iframe src="{{ overview_html }}" class="dashboard-iframe"></iframe>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Distributed Testing Framework - Comprehensive Monitoring Dashboard</p>
    </div>
    
    <style>
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
            color: #3f51b5;
        }
        
        .metric-detail {
            color: #757575;
        }
    </style>
</body>
</html>
""")
    
    # Create workers template
    workers_path = os.path.join(output_dir, "workers.html")
    with open(workers_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Comprehensive Monitoring Dashboard</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="/static/js/dashboard.js"></script>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
    </div>
    
    <div class="dashboard-grid">
        <div class="sidebar">
            <h2>Navigation</h2>
            <ul class="nav-list">
                <li><a href="/">Dashboard Home</a></li>
                
                <li class="nav-header">System Status</li>
                <li><a href="/pages/status">System Status</a></li>
                <li><a href="/pages/workers" class="active">Worker Status</a></li>
                <li><a href="/pages/tasks">Task Status</a></li>
                
                <li class="nav-header">Visualizations</li>
                <li><a href="/visualizations/system_overview">System Overview</a></li>
                <li><a href="/visualizations/worker_performance">Worker Performance</a></li>
                <li><a href="/visualizations/task_performance">Task Performance</a></li>
                <li><a href="/visualizations/resource_utilization">Resource Utilization</a></li>
            </ul>
        </div>
        
        <div class="main-content">
            <div class="dashboard-section">
                <h2>Worker Performance</h2>
                <iframe src="{{ performance_html }}" class="dashboard-iframe" data-vis-id="worker_performance"></iframe>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Distributed Testing Framework - Comprehensive Monitoring Dashboard</p>
    </div>
</body>
</html>
""")
    
    # Create tasks template
    tasks_path = os.path.join(output_dir, "tasks.html")
    with open(tasks_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Comprehensive Monitoring Dashboard</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="/static/js/dashboard.js"></script>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
    </div>
    
    <div class="dashboard-grid">
        <div class="sidebar">
            <h2>Navigation</h2>
            <ul class="nav-list">
                <li><a href="/">Dashboard Home</a></li>
                
                <li class="nav-header">System Status</li>
                <li><a href="/pages/status">System Status</a></li>
                <li><a href="/pages/workers">Worker Status</a></li>
                <li><a href="/pages/tasks" class="active">Task Status</a></li>
                
                <li class="nav-header">Visualizations</li>
                <li><a href="/visualizations/system_overview">System Overview</a></li>
                <li><a href="/visualizations/worker_performance">Worker Performance</a></li>
                <li><a href="/visualizations/task_performance">Task Performance</a></li>
                <li><a href="/visualizations/resource_utilization">Resource Utilization</a></li>
            </ul>
        </div>
        
        <div class="main-content">
            <div class="dashboard-section">
                <h2>Task Performance</h2>
                <iframe src="{{ performance_html }}" class="dashboard-iframe" data-vis-id="task_performance"></iframe>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Distributed Testing Framework - Comprehensive Monitoring Dashboard</p>
    </div>
</body>
</html>
""")
    
    # Create visualization template
    visualization_path = os.path.join(output_dir, "visualization.html")
    with open(visualization_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Comprehensive Monitoring Dashboard</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="/static/js/dashboard.js"></script>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
    </div>
    
    <div class="dashboard-grid">
        <div class="sidebar">
            <h2>Navigation</h2>
            <ul class="nav-list">
                <li><a href="/">Dashboard Home</a></li>
                
                <li class="nav-header">System Status</li>
                <li><a href="/pages/status">System Status</a></li>
                <li><a href="/pages/workers">Worker Status</a></li>
                <li><a href="/pages/tasks">Task Status</a></li>
                
                <li class="nav-header">Visualizations</li>
                <li><a href="/visualizations/system_overview" {% if vis_id == "system_overview" %}class="active"{% end %}>System Overview</a></li>
                <li><a href="/visualizations/worker_performance" {% if vis_id == "worker_performance" %}class="active"{% end %}>Worker Performance</a></li>
                <li><a href="/visualizations/task_performance" {% if vis_id == "task_performance" %}class="active"{% end %}>Task Performance</a></li>
                <li><a href="/visualizations/resource_utilization" {% if vis_id == "resource_utilization" %}class="active"{% end %}>Resource Utilization</a></li>
            </ul>
        </div>
        
        <div class="main-content">
            <div class="dashboard-section">
                <h2>{{ title }}</h2>
                <iframe src="{{ vis_html }}" class="dashboard-iframe" data-vis-id="{{ vis_id }}"></iframe>
                <div class="visualization-actions">
                    <button onclick="requestVisualization('{{ vis_id }}')">Refresh Visualization</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Distributed Testing Framework - Comprehensive Monitoring Dashboard</p>
    </div>
    
    <style>
        .visualization-actions {
            margin-top: 10px;
            text-align: right;
        }
        
        .visualization-actions button {
            padding: 8px 16px;
            background-color: #3f51b5;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .visualization-actions button:hover {
            background-color: #303f9f;
        }
    </style>
</body>
</html>
""")
    
    # Create page template
    page_path = os.path.join(output_dir, "page.html")
    with open(page_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Comprehensive Monitoring Dashboard</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="/static/js/dashboard.js"></script>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
    </div>
    
    <div class="dashboard-grid">
        <div class="sidebar">
            <h2>Navigation</h2>
            <ul class="nav-list">
                <li><a href="/">Dashboard Home</a></li>
                
                <li class="nav-header">System Status</li>
                <li><a href="/pages/status">System Status</a></li>
                <li><a href="/pages/workers">Worker Status</a></li>
                <li><a href="/pages/tasks">Task Status</a></li>
                <li><a href="/pages/fault_tolerance" {% if page_id == "fault_tolerance" %}class="active"{% end %}>Fault Tolerance</a></li>
                <li><a href="/pages/result_aggregation" {% if page_id == "result_aggregation" %}class="active"{% end %}>Result Aggregation</a></li>
                
                <li class="nav-header">Visualizations</li>
                <li><a href="/visualizations/system_overview">System Overview</a></li>
                <li><a href="/visualizations/worker_performance">Worker Performance</a></li>
                <li><a href="/visualizations/task_performance">Task Performance</a></li>
                <li><a href="/visualizations/resource_utilization">Resource Utilization</a></li>
            </ul>
        </div>
        
        <div class="main-content">
            <div class="dashboard-section">
                <h2>{{ title }}</h2>
                {% if "report_path" in page_data %}
                <iframe src="/dashboards/{{ os.path.basename(page_data['report_path']) }}" class="dashboard-iframe"></iframe>
                {% elif "html_path" in page_data %}
                <iframe src="/dashboards/{{ os.path.basename(page_data['html_path']) }}" class="dashboard-iframe"></iframe>
                {% else %}
                <div class="page-content">
                    <pre>{{ json.dumps(page_data, indent=2) }}</pre>
                </div>
                {% end %}
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Distributed Testing Framework - Comprehensive Monitoring Dashboard</p>
    </div>
    
    <style>
        .page-content {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 4px;
            overflow: auto;
        }
        
        .page-content pre {
            margin: 0;
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
</body>
</html>
""")
    
    logger.info(f"Template files created in {output_dir}")


def run_dashboard(coordinator=None, port=8888, coordinator_url=None, db_path=None):
    """
    Run the comprehensive monitoring dashboard.
    
    Args:
        coordinator: The coordinator server instance (optional)
        port: Port for the dashboard web server
        coordinator_url: URL to connect to coordinator if not directly provided
        db_path: Path to the SQLite/DuckDB database
        
    Returns:
        ComprehensiveMonitoringDashboard: Dashboard instance
    """
    # Create dashboard instance
    dashboard = ComprehensiveMonitoringDashboard(
        coordinator=coordinator,
        port=port,
        coordinator_url=coordinator_url,
        db_path=db_path
    )
    
    # Create required files
    create_css_file(dashboard.static_path)
    create_js_file(dashboard.static_path)
    create_template_files(dashboard.template_path)
    
    # Start dashboard
    dashboard.start()
    
    return dashboard


if __name__ == "__main__":
    import argparse
    import random
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Comprehensive Monitoring Dashboard")
    parser.add_argument("--port", type=int, default=8888, help="Port for dashboard web server")
    parser.add_argument("--coordinator-url", type=str, default="http://localhost:8080", help="URL of coordinator server")
    parser.add_argument("--db-path", type=str, help="Path to SQLite/DuckDB database")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run dashboard
    dashboard = run_dashboard(
        port=args.port,
        coordinator_url=args.coordinator_url,
        db_path=args.db_path
    )