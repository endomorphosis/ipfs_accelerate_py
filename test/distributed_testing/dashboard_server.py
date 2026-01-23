#!/usr/bin/env python3
"""
IPFS Accelerate Distributed Testing Framework - Dashboard Server

This script provides a web-based dashboard for monitoring the distributed testing framework,
including worker nodes, tasks, and performance metrics.

Usage:
    python dashboard_server.py --coordinator http://coordinator:8080 --port 8050
"""

import argparse
import anyio
import base64
import datetime
import io
import json
import logging
import os
import platform
import signal
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback, dash_table
from dash.exceptions import PreventUpdate
from flask import Flask, Response, jsonify, request, send_from_directory
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
UPDATE_INTERVAL = 5000  # 5 seconds
MAX_POINTS = 100       # Maximum number of points in time series charts
COLORS = {
    "primary": "#2196F3",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "danger": "#F44336",
    "info": "#00BCD4",
    "secondary": "#9E9E9E",
    "light": "#F5F5F5",
    "dark": "#212121",
    "pending": "#FFC107",
    "assigned": "#03A9F4",
    "running": "#2196F3",
    "completed": "#4CAF50",
    "failed": "#F44336",
    "cancelled": "#9E9E9E",
    "idle": "#4CAF50",
    "busy": "#2196F3",
    "offline": "#9E9E9E",
    "cpu": "#4CAF50",
    "gpu": "#2196F3",
    "memory": "#FFC107",
    "disk": "#9C27B0",
    "network": "#00BCD4",
}

# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    suppress_callback_exceptions=True,
)
app.title = "Distributed Testing Dashboard"

# Global state
coordinator_url = "http://localhost:8080"
api_key = None
last_update_time = datetime.datetime.now()
performance_data = {
    "cpu": [],
    "gpu": [],
    "memory": [],
    "tasks_completed": [],
    "tasks_failed": [],
    "worker_count": [],
}

# =====================
# Data Fetching
# =====================

async def fetch_coordinator_status():
    """Fetch status data from the coordinator."""
    global last_update_time
    
    try:
        # Create headers
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{coordinator_url}/status", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    last_update_time = datetime.datetime.now()
                    return data
                else:
                    logger.error(f"Failed to fetch status: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching status: {str(e)}")
        return None

async def fetch_worker_data():
    """Fetch worker data from the coordinator."""
    try:
        # Create headers
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{coordinator_url}/workers", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("workers", [])
                else:
                    logger.error(f"Failed to fetch workers: {response.status}")
                    return []
    except Exception as e:
        logger.error(f"Error fetching workers: {str(e)}")
        return []

async def fetch_task_data():
    """Fetch task data from the coordinator."""
    try:
        # Create headers
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{coordinator_url}/tasks", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("tasks", [])
                else:
                    logger.error(f"Failed to fetch tasks: {response.status}")
                    return []
    except Exception as e:
        logger.error(f"Error fetching tasks: {str(e)}")
        return []

async def fetch_statistics():
    """Fetch statistics data from the coordinator."""
    try:
        # Create headers
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{coordinator_url}/statistics", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to fetch statistics: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        return None

async def update_performance_data():
    """Update performance data time series."""
    global performance_data
    
    # Fetch statistics
    statistics = await fetch_statistics()
    if not statistics:
        return
    
    # Fetch worker data
    workers = await fetch_worker_data()
    
    # Current timestamp
    timestamp = datetime.datetime.now()
    
    # Update CPU data
    cpu_values = [worker.get("hardware_metrics", {}).get("cpu_percent", 0) for worker in workers if worker.get("status") != "offline"]
    cpu_avg = sum(cpu_values) / max(len(cpu_values), 1)
    performance_data["cpu"].append((timestamp, cpu_avg))
    
    # Update GPU data if available
    gpu_values = []
    for worker in workers:
        if worker.get("status") == "offline":
            continue
        
        gpu_metrics = worker.get("hardware_metrics", {}).get("gpu", [])
        if gpu_metrics:
            for gpu in gpu_metrics:
                gpu_values.append(gpu.get("memory_utilization_percent", 0))
    
    gpu_avg = sum(gpu_values) / max(len(gpu_values), 1) if gpu_values else 0
    performance_data["gpu"].append((timestamp, gpu_avg))
    
    # Update memory data
    memory_values = [worker.get("hardware_metrics", {}).get("memory_percent", 0) for worker in workers if worker.get("status") != "offline"]
    memory_avg = sum(memory_values) / max(len(memory_values), 1)
    performance_data["memory"].append((timestamp, memory_avg))
    
    # Update task data
    tasks_completed = statistics.get("tasks_completed", 0)
    tasks_failed = statistics.get("tasks_failed", 0)
    worker_count = statistics.get("workers_active", 0)
    
    performance_data["tasks_completed"].append((timestamp, tasks_completed))
    performance_data["tasks_failed"].append((timestamp, tasks_failed))
    performance_data["worker_count"].append((timestamp, worker_count))
    
    # Limit the number of data points
    for key in performance_data:
        performance_data[key] = performance_data[key][-MAX_POINTS:]

async def data_update_loop():
    """Background task to periodically update data."""
    while True:
        await update_performance_data()
        await anyio.sleep(UPDATE_INTERVAL / 1000)

# =====================
# Dashboard Layout
# =====================

def create_header():
    """Create the dashboard header."""
    return dbc.Row(
        [
            dbc.Col(
                html.H1("Distributed Testing Dashboard", className="text-primary"),
                width=8,
            ),
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Span("Last Updated: ", className="me-2"),
                            html.Span(id="last-update-time", className="fw-bold"),
                        ],
                        className="text-end mt-2",
                    ),
                    html.Div(
                        [
                            html.Span("Coordinator: ", className="me-2"),
                            html.Span(id="coordinator-status", className="fw-bold"),
                        ],
                        className="text-end",
                    ),
                ],
                width=4,
            ),
        ],
        className="mb-4 mt-4",
    )

def create_status_cards():
    """Create status summary cards."""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Active Workers", className="text-white bg-primary"),
                        dbc.CardBody(
                            [
                                html.H2(id="active-workers-count", className="card-text text-center"),
                                html.P("Connected worker nodes", className="card-text text-center text-muted small"),
                            ]
                        ),
                    ],
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Pending Tasks", className="text-white bg-warning"),
                        dbc.CardBody(
                            [
                                html.H2(id="pending-tasks-count", className="card-text text-center"),
                                html.P("Tasks waiting to be assigned", className="card-text text-center text-muted small"),
                            ]
                        ),
                    ],
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Running Tasks", className="text-white bg-info"),
                        dbc.CardBody(
                            [
                                html.H2(id="running-tasks-count", className="card-text text-center"),
                                html.P("Tasks currently executing", className="card-text text-center text-muted small"),
                            ]
                        ),
                    ],
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Completed Tasks", className="text-white bg-success"),
                        dbc.CardBody(
                            [
                                html.H2(id="completed-tasks-count", className="card-text text-center"),
                                html.P("Successfully completed tasks", className="card-text text-center text-muted small"),
                            ]
                        ),
                    ],
                ),
                width=3,
            ),
        ],
        className="mb-4",
    )

def create_resource_usage_charts():
    """Create resource usage charts."""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Resource Usage", className="text-white bg-primary"),
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="resource-usage-chart",
                                    style={"height": "300px"},
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                    ],
                ),
                width=12,
            ),
        ],
        className="mb-4",
    )

def create_worker_status_section():
    """Create worker status section."""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Worker Nodes", className="text-white bg-primary"),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.InputGroup(
                                                    [
                                                        dbc.InputGroupText("Filter:"),
                                                        dbc.Input(id="worker-filter-input", placeholder="Search workers..."),
                                                    ],
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.InputGroup(
                                                    [
                                                        dbc.InputGroupText("Status:"),
                                                        dbc.Select(
                                                            id="worker-status-filter",
                                                            options=[
                                                                {"label": "All", "value": "all"},
                                                                {"label": "Idle", "value": "idle"},
                                                                {"label": "Busy", "value": "busy"},
                                                                {"label": "Offline", "value": "offline"},
                                                            ],
                                                            value="all",
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                html.Div(id="worker-table-container", style={"maxHeight": "400px", "overflow": "auto"}),
                            ]
                        ),
                    ],
                ),
                width=12,
            ),
        ],
        className="mb-4",
    )

def create_task_queue_section():
    """Create task queue section."""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Task Queue", className="text-white bg-primary"),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.InputGroup(
                                                    [
                                                        dbc.InputGroupText("Filter:"),
                                                        dbc.Input(id="task-filter-input", placeholder="Search tasks..."),
                                                    ],
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.InputGroup(
                                                    [
                                                        dbc.InputGroupText("Status:"),
                                                        dbc.Select(
                                                            id="task-status-filter",
                                                            options=[
                                                                {"label": "All", "value": "all"},
                                                                {"label": "Pending", "value": "pending"},
                                                                {"label": "Assigned", "value": "assigned"},
                                                                {"label": "Running", "value": "running"},
                                                                {"label": "Completed", "value": "completed"},
                                                                {"label": "Failed", "value": "failed"},
                                                                {"label": "Cancelled", "value": "cancelled"},
                                                            ],
                                                            value="all",
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                html.Div(id="task-table-container", style={"maxHeight": "400px", "overflow": "auto"}),
                            ]
                        ),
                    ],
                ),
                width=12,
            ),
        ],
        className="mb-4",
    )

def create_performance_charts():
    """Create performance charts section."""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Task Completion Rate", className="text-white bg-primary"),
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="task-completion-chart",
                                    style={"height": "300px"},
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                    ],
                ),
                width=6,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Worker Count", className="text-white bg-primary"),
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="worker-count-chart",
                                    style={"height": "300px"},
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                    ],
                ),
                width=6,
            ),
        ],
        className="mb-4",
    )

# Build the layout
app.layout = dbc.Container(
    [
        dcc.Interval(
            id="interval-component",
            interval=UPDATE_INTERVAL,  # in milliseconds
            n_intervals=0,
        ),
        create_header(),
        create_status_cards(),
        create_resource_usage_charts(),
        create_worker_status_section(),
        create_task_queue_section(),
        create_performance_charts(),
    ],
    fluid=True,
    className="px-4 py-3 bg-light",
)

# =====================
# Callbacks
# =====================

@app.callback(
    [
        Output("last-update-time", "children"),
        Output("coordinator-status", "children"),
        Output("coordinator-status", "className"),
        Output("active-workers-count", "children"),
        Output("pending-tasks-count", "children"),
        Output("running-tasks-count", "children"),
        Output("completed-tasks-count", "children"),
    ],
    [Input("interval-component", "n_intervals")],
)
async def update_status_indicators(n):
    """Update status indicators."""
    # Format last update time
    last_update_str = last_update_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Fetch status data
    status_data = await fetch_coordinator_status()
    
    if not status_data:
        return (
            last_update_str,
            "Disconnected",
            "text-danger fw-bold",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        )
    
    # Extract values from status data
    active_workers = status_data.get("statistics", {}).get("workers_active", 0)
    pending_tasks = status_data.get("statistics", {}).get("tasks_pending", 0)
    running_tasks = status_data.get("statistics", {}).get("tasks_running", 0)
    completed_tasks = status_data.get("statistics", {}).get("tasks_completed", 0)
    
    return (
        last_update_str,
        "Connected",
        "text-success fw-bold",
        active_workers,
        pending_tasks,
        running_tasks,
        completed_tasks,
    )

@app.callback(
    Output("resource-usage-chart", "figure"),
    [Input("interval-component", "n_intervals")],
)
async def update_resource_usage_chart(n):
    """Update resource usage chart."""
    # Create traces
    cpu_data = performance_data["cpu"]
    memory_data = performance_data["memory"]
    gpu_data = performance_data["gpu"]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    if cpu_data:
        times, values = zip(*cpu_data)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name="CPU Usage (%)",
                line=dict(color=COLORS["cpu"], width=2),
            )
        )
    
    if memory_data:
        times, values = zip(*memory_data)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name="Memory Usage (%)",
                line=dict(color=COLORS["memory"], width=2),
            )
        )
    
    if gpu_data and any(v > 0 for _, v in gpu_data):
        times, values = zip(*gpu_data)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name="GPU Memory (%)",
                line=dict(color=COLORS["gpu"], width=2),
            )
        )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(211, 211, 211, 0.5)",
        ),
        yaxis=dict(
            title="Usage (%)",
            showgrid=True,
            gridcolor="rgba(211, 211, 211, 0.5)",
            range=[0, 100],
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    
    return fig

@app.callback(
    Output("worker-table-container", "children"),
    [
        Input("interval-component", "n_intervals"),
        Input("worker-filter-input", "value"),
        Input("worker-status-filter", "value"),
    ],
)
async def update_worker_table(n, filter_value, status_filter):
    """Update worker table."""
    # Fetch worker data
    workers = await fetch_worker_data()
    
    if not workers:
        return html.Div("No worker data available", className="text-muted text-center py-3")
    
    # Apply filters
    filtered_workers = workers
    
    if filter_value:
        filter_value = filter_value.lower()
        filtered_workers = [
            w for w in filtered_workers
            if filter_value in w.get("id", "").lower() or
               filter_value in w.get("hostname", "").lower() or
               filter_value in str(w.get("capabilities", {})).lower()
        ]
    
    if status_filter and status_filter != "all":
        filtered_workers = [
            w for w in filtered_workers
            if w.get("status", "").lower() == status_filter.lower()
        ]
    
    # Prepare table data
    table_data = []
    
    for worker in filtered_workers:
        worker_id = worker.get("id", "")
        hostname = worker.get("hostname", "")
        status = worker.get("status", "unknown")
        capabilities = worker.get("capabilities", {})
        hardware_metrics = worker.get("hardware_metrics", {})
        current_task_id = worker.get("current_task_id")
        
        hardware_list = []
        
        # CPU info
        if "cpu" in capabilities:
            cpu_info = capabilities.get("cpu", {})
            cores = cpu_info.get("cores", 0)
            threads = cpu_info.get("threads", 0)
            hardware_list.append(f"CPU: {cores} cores/{threads} threads")
        
        # GPU info
        if "gpu" in capabilities:
            gpu_info = capabilities.get("gpu", {})
            count = gpu_info.get("count", 0)
            name = gpu_info.get("name", "Unknown GPU")
            memory = gpu_info.get("memory_gb", 0)
            hardware_list.append(f"GPU: {count}x {name} ({memory} GB)")
        
        # Memory info
        if "memory" in capabilities:
            memory_info = capabilities.get("memory", {})
            total = memory_info.get("total_gb", 0)
            hardware_list.append(f"Memory: {total} GB")
        
        # Task info
        task_info = "None" if not current_task_id else current_task_id
        
        # Usage info
        cpu_usage = hardware_metrics.get("cpu_percent", 0)
        memory_usage = hardware_metrics.get("memory_percent", 0)
        
        # Add to table data
        table_data.append({
            "id": worker_id[:8] + "...",
            "hostname": hostname,
            "status": status,
            "hardware": ", ".join(hardware_list),
            "task": task_info,
            "cpu": f"{cpu_usage:.1f}%",
            "memory": f"{memory_usage:.1f}%",
        })
    
    # Create table
    if table_data:
        return dash_table.DataTable(
            id="worker-table",
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Hostname", "id": "hostname"},
                {"name": "Status", "id": "status"},
                {"name": "Hardware", "id": "hardware"},
                {"name": "Current Task", "id": "task"},
                {"name": "CPU", "id": "cpu"},
                {"name": "Memory", "id": "memory"},
            ],
            data=table_data,
            style_table={"minWidth": "100%"},
            style_cell={
                "textAlign": "left",
                "padding": "5px",
                "fontSize": "12px",
            },
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'idle'"},
                    "backgroundColor": "rgba(76, 175, 80, 0.2)",
                },
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'busy'"},
                    "backgroundColor": "rgba(33, 150, 243, 0.2)",
                },
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'offline'"},
                    "backgroundColor": "rgba(158, 158, 158, 0.2)",
                },
            ],
        )
    else:
        return html.Div("No workers match the current filters", className="text-muted text-center py-3")

@app.callback(
    Output("task-table-container", "children"),
    [
        Input("interval-component", "n_intervals"),
        Input("task-filter-input", "value"),
        Input("task-status-filter", "value"),
    ],
)
async def update_task_table(n, filter_value, status_filter):
    """Update task table."""
    # Fetch task data
    tasks = await fetch_task_data()
    
    if not tasks:
        return html.Div("No task data available", className="text-muted text-center py-3")
    
    # Apply filters
    filtered_tasks = tasks
    
    if filter_value:
        filter_value = filter_value.lower()
        filtered_tasks = [
            t for t in filtered_tasks
            if filter_value in t.get("id", "").lower() or
               filter_value in t.get("test_path", "").lower() or
               filter_value in str(t.get("parameters", {})).lower()
        ]
    
    if status_filter and status_filter != "all":
        filtered_tasks = [
            t for t in filtered_tasks
            if t.get("status", "").lower() == status_filter.lower()
        ]
    
    # Prepare table data
    table_data = []
    
    for task in filtered_tasks:
        task_id = task.get("id", "")
        test_path = task.get("test_path", "")
        status = task.get("status", "unknown")
        parameters = task.get("parameters", {})
        worker_id = task.get("worker_id", "")
        assigned_time = task.get("assigned_time")
        start_time = task.get("start_time")
        end_time = task.get("end_time")
        
        # Format times
        assigned_time_str = format_timestamp(assigned_time) if assigned_time else "N/A"
        start_time_str = format_timestamp(start_time) if start_time else "N/A"
        end_time_str = format_timestamp(end_time) if end_time else "N/A"
        
        # Calculate duration
        duration = "N/A"
        if start_time and end_time:
            duration = f"{end_time - start_time:.2f}s"
        
        # Format parameters
        param_str = ", ".join([f"{k}: {v}" for k, v in parameters.items()])
        
        # Add to table data
        table_data.append({
            "id": task_id[:8] + "...",
            "test_path": test_path,
            "status": status,
            "parameters": param_str[:30] + "..." if len(param_str) > 30 else param_str,
            "worker": worker_id[:8] + "..." if worker_id else "N/A",
            "assigned": assigned_time_str,
            "duration": duration,
        })
    
    # Sort tasks by status (pending and running first)
    status_order = {"pending": 0, "assigned": 1, "running": 2, "completed": 3, "failed": 4, "cancelled": 5}
    table_data.sort(key=lambda x: status_order.get(x["status"].lower(), 99))
    
    # Create table
    if table_data:
        return dash_table.DataTable(
            id="task-table",
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Test Path", "id": "test_path"},
                {"name": "Status", "id": "status"},
                {"name": "Parameters", "id": "parameters"},
                {"name": "Worker", "id": "worker"},
                {"name": "Assigned", "id": "assigned"},
                {"name": "Duration", "id": "duration"},
            ],
            data=table_data,
            style_table={"minWidth": "100%"},
            style_cell={
                "textAlign": "left",
                "padding": "5px",
                "fontSize": "12px",
            },
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'pending'"},
                    "backgroundColor": "rgba(255, 193, 7, 0.2)",
                },
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'assigned'"},
                    "backgroundColor": "rgba(3, 169, 244, 0.2)",
                },
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'running'"},
                    "backgroundColor": "rgba(33, 150, 243, 0.2)",
                },
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'completed'"},
                    "backgroundColor": "rgba(76, 175, 80, 0.2)",
                },
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'failed'"},
                    "backgroundColor": "rgba(244, 67, 54, 0.2)",
                },
                {
                    "if": {"column_id": "status", "filter_query": "{status} eq 'cancelled'"},
                    "backgroundColor": "rgba(158, 158, 158, 0.2)",
                },
            ],
        )
    else:
        return html.Div("No tasks match the current filters", className="text-muted text-center py-3")

@app.callback(
    Output("task-completion-chart", "figure"),
    [Input("interval-component", "n_intervals")],
)
async def update_task_completion_chart(n):
    """Update task completion chart."""
    # Create traces
    completed_data = performance_data["tasks_completed"]
    failed_data = performance_data["tasks_failed"]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    if completed_data:
        times, values = zip(*completed_data)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name="Completed Tasks",
                line=dict(color=COLORS["success"], width=2),
            )
        )
    
    if failed_data:
        times, values = zip(*failed_data)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name="Failed Tasks",
                line=dict(color=COLORS["danger"], width=2),
            )
        )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(211, 211, 211, 0.5)",
        ),
        yaxis=dict(
            title="Count",
            showgrid=True,
            gridcolor="rgba(211, 211, 211, 0.5)",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    
    return fig

@app.callback(
    Output("worker-count-chart", "figure"),
    [Input("interval-component", "n_intervals")],
)
async def update_worker_count_chart(n):
    """Update worker count chart."""
    # Create traces
    worker_data = performance_data["worker_count"]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    if worker_data:
        times, values = zip(*worker_data)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name="Active Workers",
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy",
            )
        )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(211, 211, 211, 0.5)",
        ),
        yaxis=dict(
            title="Count",
            showgrid=True,
            gridcolor="rgba(211, 211, 211, 0.5)",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    
    return fig

def format_timestamp(timestamp):
    """Format a timestamp for display."""
    if not timestamp:
        return "N/A"
    
    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(timestamp)

# =====================
# Main Entrypoint
# =====================

async def start_background_tasks():
    """Start background tasks."""
    # TODO: Replace with task group - asyncio.create_task(data_update_loop())

def main():
    """Main entrypoint."""
    global coordinator_url, api_key
    
    parser = argparse.ArgumentParser(description="Distributed Testing Framework Dashboard")
    parser.add_argument("--coordinator", default="http://localhost:8080", help="URL of the coordinator server")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--api-key", help="API key for authentication with coordinator")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--open-browser", action="store_true", help="Open browser after starting")
    
    args = parser.parse_args()
    
    coordinator_url = args.coordinator
    api_key = args.api_key
    
    # Print startup message
    print(f"Starting dashboard at http://localhost:{args.port}")
    print(f"Coordinator URL: {coordinator_url}")
    
    # Run app
    if args.open_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    
    app.run_server(
        host="0.0.0.0",
        port=args.port,
        debug=args.debug,
        dev_tools_hot_reload=args.debug,
    )

if __name__ == "__main__":
    main()