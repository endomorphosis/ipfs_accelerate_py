#!/usr/bin/env python3
"""
Distributed Testing Framework - Monitoring Dashboard Server

This module implements a simple web server that provides a dashboard for monitoring
the distributed testing framework, including:

- Real-time worker status visualization
- Task execution status and history
- System health metrics and alerts
- Performance metrics and statistics
- Resource utilization graphs
- Test result summaries
- Failure analysis and debugging
- Worker distribution visualization

Usage:
    python dashboard_server.py --host 0.0.0.0 --port 8081 --coordinator-url http://localhost:8080
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import socket
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_server")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests not available. Using urllib for HTTP requests.")
    REQUESTS_AVAILABLE = False
    import urllib.request
    import urllib.error

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    logger.warning("websocket-client not available. WebSocket monitoring disabled.")
    WEBSOCKET_AVAILABLE = False

# HTML template for dashboard
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distributed Testing Framework Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            text-align: center;
        }
        h1, h2, h3 {
            margin: 0;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            position: relative;
        }
        .status-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-healthy {
            background-color: #2ecc71;
            color: white;
        }
        .badge-warning {
            background-color: #f39c12;
            color: white;
        }
        .badge-critical {
            background-color: #e74c3c;
            color: white;
        }
        .badge-unknown {
            background-color: #95a5a6;
            color: white;
        }
        .worker-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .worker-card {
            background-color: #f9f9f9;
            border-radius: 6px;
            padding: 15px;
            border-left: 4px solid #3498db;
        }
        .worker-card.healthy {
            border-left-color: #2ecc71;
        }
        .worker-card.warning {
            border-left-color: #f39c12;
        }
        .worker-card.critical {
            border-left-color: #e74c3c;
        }
        .worker-card.disconnected {
            border-left-color: #95a5a6;
            opacity: 0.7;
        }
        .worker-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metrics {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 13px;
        }
        .metric {
            text-align: center;
        }
        .metric-value {
            font-weight: bold;
            font-size: 15px;
        }
        .task-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
            max-height: 300px;
            overflow-y: auto;
        }
        .task-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .task-id {
            font-weight: bold;
            flex: 1;
        }
        .task-status {
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        .status-queued {
            background-color: #3498db;
            color: white;
        }
        .status-running {
            background-color: #2ecc71;
            color: white;
        }
        .status-completed {
            background-color: #27ae60;
            color: white;
        }
        .status-failed {
            background-color: #e74c3c;
            color: white;
        }
        .chart-container {
            height: 300px;
            width: 100%;
        }
        .refresh-info {
            text-align: right;
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .issues-list {
            color: #e74c3c;
            padding-left: 20px;
            margin: 5px 0;
            font-size: 13px;
        }
        .progress-container {
            margin-top: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            height: 8px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background-color: #3498db;
        }
        #system-summary {
            grid-column: 1 / -1;
            background-color: #2c3e50;
            color: white;
        }
        #system-summary .metrics {
            justify-content: space-around;
            margin-top: 15px;
        }
        #system-summary .metric-value {
            font-size: 24px;
        }
        .auto-refresh {
            margin-top: 10px;
            text-align: right;
        }
        .tabs {
            margin-top: 20px;
            display: flex;
            border-bottom: 1px solid #ccc;
        }
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            background-color: #f5f5f5;
            border: 1px solid #ccc;
            border-bottom: none;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }
        .tab.active {
            background-color: white;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
        }
        .tab-content {
            display: none;
            padding-top: 20px;
        }
        .tab-content.active {
            display: block;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f5f5f5;
        }
        .alert {
            background-color: #ffecb3;
            border-left: 4px solid #f39c12;
            padding: 10px 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .alert-critical {
            background-color: #ffcccc;
            border-left-color: #e74c3c;
        }
        .timestamp {
            font-size: 12px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <header>
        <h1>Distributed Testing Framework Dashboard</h1>
        <div id="connection-status">Connecting to coordinator...</div>
    </header>
    
    <div class="container">
        <div class="tabs">
            <div class="tab active" data-tab="overview">Overview</div>
            <div class="tab" data-tab="workers">Workers</div>
            <div class="tab" data-tab="tasks">Tasks</div>
            <div class="tab" data-tab="performance">Performance</div>
            <div class="tab" data-tab="alerts">Alerts</div>
        </div>
        
        <div id="overview" class="tab-content active">
            <div class="dashboard-grid">
                <div id="system-summary" class="card">
                    <h2>System Summary</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div>Total Workers</div>
                            <div id="total-workers" class="metric-value">0</div>
                        </div>
                        <div class="metric">
                            <div>Active Workers</div>
                            <div id="active-workers" class="metric-value">0</div>
                        </div>
                        <div class="metric">
                            <div>Running Tasks</div>
                            <div id="running-tasks" class="metric-value">0</div>
                        </div>
                        <div class="metric">
                            <div>Queued Tasks</div>
                            <div id="queued-tasks" class="metric-value">0</div>
                        </div>
                        <div class="metric">
                            <div>Completed Tasks</div>
                            <div id="completed-tasks" class="metric-value">0</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Worker Status</h3>
                    <div class="status-badge badge-healthy" id="workers-status-badge">Healthy</div>
                    <div id="worker-stats-container">
                        <div class="metrics">
                            <div class="metric">
                                <div>Healthy</div>
                                <div id="healthy-workers" class="metric-value">0</div>
                            </div>
                            <div class="metric">
                                <div>Warning</div>
                                <div id="warning-workers" class="metric-value">0</div>
                            </div>
                            <div class="metric">
                                <div>Critical</div>
                                <div id="critical-workers" class="metric-value">0</div>
                            </div>
                            <div class="metric">
                                <div>Disconnected</div>
                                <div id="disconnected-workers" class="metric-value">0</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Task Status</h3>
                    <div id="task-stats-container">
                        <div class="progress-container">
                            <div id="tasks-progress" class="progress-bar" style="width: 0%"></div>
                        </div>
                        <div class="metrics">
                            <div class="metric">
                                <div>Success Rate</div>
                                <div id="task-success-rate" class="metric-value">0%</div>
                            </div>
                            <div class="metric">
                                <div>Avg. Duration</div>
                                <div id="task-avg-duration" class="metric-value">0s</div>
                            </div>
                            <div class="metric">
                                <div>Failures</div>
                                <div id="task-failures" class="metric-value">0</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Recent Alerts</h3>
                    <div id="recent-alerts-container">
                        <div id="alerts-placeholder">No recent alerts</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Active Workers</h3>
                    <div id="active-workers-list" class="worker-container">
                        <div class="worker-card">
                            <div class="worker-name">No workers connected</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Recent Tasks</h3>
                    <div id="recent-tasks-container">
                        <ul id="recent-tasks" class="task-list">
                            <li class="task-item">
                                <span class="task-id">No recent tasks</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="workers" class="tab-content">
            <h2>Worker Details</h2>
            <div id="worker-details-container" class="dashboard-grid">
                <!-- Worker details will be populated here -->
                <div class="card">
                    <h3>No workers connected</h3>
                </div>
            </div>
        </div>
        
        <div id="tasks" class="tab-content">
            <h2>Task Management</h2>
            <div class="dashboard-grid">
                <div class="card">
                    <h3>Task Queue</h3>
                    <div id="task-queue-container">
                        <ul id="task-queue" class="task-list">
                            <li class="task-item">
                                <span class="task-id">No tasks in queue</span>
                            </li>
                        </ul>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Running Tasks</h3>
                    <div id="running-tasks-container">
                        <ul id="running-tasks-list" class="task-list">
                            <li class="task-item">
                                <span class="task-id">No running tasks</span>
                            </li>
                        </ul>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Completed Tasks</h3>
                    <div id="completed-tasks-container">
                        <ul id="completed-tasks-list" class="task-list">
                            <li class="task-item">
                                <span class="task-id">No completed tasks</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="performance" class="tab-content">
            <h2>Performance Metrics</h2>
            <div class="dashboard-grid">
                <div class="card">
                    <h3>Worker Performance</h3>
                    <div id="worker-performance-container">
                        <table id="worker-performance-table">
                            <thead>
                                <tr>
                                    <th>Worker</th>
                                    <th>Tasks Completed</th>
                                    <th>Success Rate</th>
                                    <th>Avg. Time</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td colspan="5">No data available</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Task Type Performance</h3>
                    <div id="task-type-performance-container">
                        <table id="task-type-performance-table">
                            <thead>
                                <tr>
                                    <th>Task Type</th>
                                    <th>Count</th>
                                    <th>Success Rate</th>
                                    <th>Avg. Time</th>
                                    <th>Best Worker</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td colspan="5">No data available</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="alerts" class="tab-content">
            <h2>System Alerts</h2>
            <div id="all-alerts-container">
                <div class="alert">
                    <strong>No alerts to display</strong>
                </div>
            </div>
        </div>
        
        <div class="auto-refresh">
            <label>
                <input type="checkbox" id="auto-refresh" checked> Auto-refresh (10s)
            </label>
        </div>
        
        <div class="refresh-info">
            Last updated: <span id="last-updated">Never</span>
        </div>
    </div>
    
    <script>
        // Variables
        let autoRefresh = true;
        let refreshInterval = 10000; // 10 seconds
        let refreshTimer;
        let lastData = {};
        
        // Tabs functionality
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs and content
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                tab.classList.add('active');
                const tabId = tab.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
        
        // Auto-refresh toggle
        document.getElementById('auto-refresh').addEventListener('change', function() {
            autoRefresh = this.checked;
            if (autoRefresh) {
                startRefreshTimer();
            } else {
                clearTimeout(refreshTimer);
            }
        });
        
        // Initial data fetch
        fetchDashboardData();
        
        function startRefreshTimer() {
            clearTimeout(refreshTimer);
            refreshTimer = setTimeout(() => {
                if (autoRefresh) {
                    fetchDashboardData();
                }
            }, refreshInterval);
        }
        
        function fetchDashboardData() {
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    lastData = data;
                    updateDashboard(data);
                    document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                    document.getElementById('connection-status').textContent = 'Connected to coordinator';
                    document.getElementById('connection-status').style.color = '#2ecc71';
                    startRefreshTimer();
                })
                .catch(error => {
                    console.error('Error fetching dashboard data:', error);
                    document.getElementById('connection-status').textContent = 'Error connecting to coordinator';
                    document.getElementById('connection-status').style.color = '#e74c3c';
                    startRefreshTimer();
                });
        }
        
        function updateDashboard(data) {
            // Update system summary
            document.getElementById('total-workers').textContent = data.workers.total || 0;
            document.getElementById('active-workers').textContent = data.workers.active || 0;
            document.getElementById('running-tasks').textContent = data.tasks.running || 0;
            document.getElementById('queued-tasks').textContent = data.tasks.queued || 0;
            document.getElementById('completed-tasks').textContent = data.tasks.completed || 0;
            
            // Update worker status
            document.getElementById('healthy-workers').textContent = data.workers.healthy || 0;
            document.getElementById('warning-workers').textContent = data.workers.warning || 0;
            document.getElementById('critical-workers').textContent = data.workers.critical || 0;
            document.getElementById('disconnected-workers').textContent = data.workers.disconnected || 0;
            
            // Update worker status badge
            const workersBadge = document.getElementById('workers-status-badge');
            if (data.workers.critical > 0) {
                workersBadge.textContent = 'Critical';
                workersBadge.className = 'status-badge badge-critical';
            } else if (data.workers.warning > 0) {
                workersBadge.textContent = 'Warning';
                workersBadge.className = 'status-badge badge-warning';
            } else if (data.workers.active > 0) {
                workersBadge.textContent = 'Healthy';
                workersBadge.className = 'status-badge badge-healthy';
            } else {
                workersBadge.textContent = 'No Workers';
                workersBadge.className = 'status-badge badge-unknown';
            }
            
            // Update task status
            const totalTasks = data.tasks.completed + data.tasks.failed;
            const successRate = totalTasks > 0 ? Math.round((data.tasks.completed / totalTasks) * 100) : 0;
            document.getElementById('task-success-rate').textContent = successRate + '%';
            document.getElementById('task-failures').textContent = data.tasks.failed || 0;
            document.getElementById('task-avg-duration').textContent = data.tasks.avg_duration ? data.tasks.avg_duration + 's' : '0s';
            
            // Update tasks progress bar
            const progressPercent = totalTasks > 0 ? (data.tasks.completed / totalTasks) * 100 : 0;
            document.getElementById('tasks-progress').style.width = progressPercent + '%';
            
            // Update active workers list
            updateActiveWorkersList(data.worker_details);
            
            // Update worker details
            updateWorkerDetails(data.worker_details);
            
            // Update task lists
            updateTaskLists(data.recent_tasks, data.queued_tasks, data.running_tasks_details, data.completed_tasks);
            
            // Update alerts
            updateAlerts(data.alerts);
            
            // Update performance tables
            updatePerformanceTables(data.worker_performance, data.task_type_performance);
        }
        
        function updateActiveWorkersList(workers) {
            const container = document.getElementById('active-workers-list');
            container.innerHTML = '';
            
            if (!workers || Object.keys(workers).length === 0) {
                container.innerHTML = '<div class="worker-card"><div class="worker-name">No workers connected</div></div>';
                return;
            }
            
            // Filter to show only active workers
            const activeWorkers = Object.entries(workers).filter(([_, worker]) => 
                worker.status === 'active' || worker.status === 'busy'
            );
            
            if (activeWorkers.length === 0) {
                container.innerHTML = '<div class="worker-card"><div class="worker-name">No active workers</div></div>';
                return;
            }
            
            activeWorkers.forEach(([workerId, worker]) => {
                const workerCard = document.createElement('div');
                workerCard.className = `worker-card ${worker.health_status || 'unknown'}`;
                
                let taskCount = worker.running_tasks || 0;
                let cpuMetric = worker.metrics && worker.metrics.cpu_percent ? 
                    `${Math.round(worker.metrics.cpu_percent)}%` : 'N/A';
                let memoryMetric = worker.metrics && worker.metrics.memory_available_gb ? 
                    `${worker.metrics.memory_available_gb.toFixed(1)} GB` : 'N/A';
                
                workerCard.innerHTML = `
                    <div class="worker-name">${workerId}</div>
                    <div>${worker.hostname || 'Unknown host'}</div>
                    <div class="metrics">
                        <div class="metric">
                            <div>Tasks</div>
                            <div class="metric-value">${taskCount}</div>
                        </div>
                        <div class="metric">
                            <div>CPU</div>
                            <div class="metric-value">${cpuMetric}</div>
                        </div>
                        <div class="metric">
                            <div>Mem</div>
                            <div class="metric-value">${memoryMetric}</div>
                        </div>
                    </div>
                `;
                
                container.appendChild(workerCard);
            });
        }
        
        function updateWorkerDetails(workers) {
            const container = document.getElementById('worker-details-container');
            container.innerHTML = '';
            
            if (!workers || Object.keys(workers).length === 0) {
                container.innerHTML = '<div class="card"><h3>No workers connected</h3></div>';
                return;
            }
            
            Object.entries(workers).forEach(([workerId, worker]) => {
                const card = document.createElement('div');
                card.className = 'card';
                
                let healthClass = 'badge-unknown';
                if (worker.health_status === 'healthy') healthClass = 'badge-healthy';
                else if (worker.health_status === 'warning') healthClass = 'badge-warning';
                else if (worker.health_status === 'critical') healthClass = 'badge-critical';
                
                let issuesHtml = '';
                if (worker.issues && worker.issues.length > 0) {
                    issuesHtml = '<ul class="issues-list">' + 
                        worker.issues.map(issue => `<li>${issue}</li>`).join('') + 
                        '</ul>';
                }
                
                card.innerHTML = `
                    <h3>${workerId}</h3>
                    <div class="status-badge ${healthClass}">${worker.health_status || 'Unknown'}</div>
                    <p><strong>Hostname:</strong> ${worker.hostname || 'Unknown'}</p>
                    <p><strong>Status:</strong> ${worker.status || 'Unknown'}</p>
                    <p><strong>Connected since:</strong> ${formatDate(worker.registration_time)}</p>
                    ${issuesHtml}
                    <div class="metrics">
                        <div class="metric">
                            <div>Tasks Running</div>
                            <div class="metric-value">${worker.running_tasks || 0}</div>
                        </div>
                        <div class="metric">
                            <div>Tasks Completed</div>
                            <div class="metric-value">${worker.completed_tasks || 0}</div>
                        </div>
                        <div class="metric">
                            <div>Success Rate</div>
                            <div class="metric-value">${worker.success_rate ? (worker.success_rate * 100).toFixed(1) + '%' : 'N/A'}</div>
                        </div>
                    </div>
                `;
                
                container.appendChild(card);
            });
        }
        
        function updateTaskLists(recentTasks, queuedTasks, runningTasks, completedTasks) {
            // Recent tasks on dashboard
            updateTaskList('recent-tasks', recentTasks);
            
            // Task queue
            updateTaskList('task-queue', queuedTasks);
            
            // Running tasks
            updateTaskList('running-tasks-list', runningTasks);
            
            // Completed tasks
            updateTaskList('completed-tasks-list', completedTasks);
        }
        
        function updateTaskList(elementId, tasks) {
            const listElement = document.getElementById(elementId);
            listElement.innerHTML = '';
            
            if (!tasks || tasks.length === 0) {
                const emptyItem = document.createElement('li');
                emptyItem.className = 'task-item';
                emptyItem.innerHTML = `<span class="task-id">No tasks to display</span>`;
                listElement.appendChild(emptyItem);
                return;
            }
            
            tasks.forEach(task => {
                const item = document.createElement('li');
                item.className = 'task-item';
                
                let statusClass = '';
                if (task.status === 'queued') statusClass = 'status-queued';
                else if (task.status === 'running') statusClass = 'status-running';
                else if (task.status === 'completed') statusClass = 'status-completed';
                else if (task.status === 'failed') statusClass = 'status-failed';
                
                item.innerHTML = `
                    <span class="task-id">${task.task_id}</span>
                    <span class="task-type">${task.type || 'Unknown'}</span>
                    <span class="task-status ${statusClass}">${task.status || 'Unknown'}</span>
                `;
                
                listElement.appendChild(item);
            });
        }
        
        function updateAlerts(alerts) {
            // Recent alerts on dashboard
            const recentAlertsContainer = document.getElementById('recent-alerts-container');
            recentAlertsContainer.innerHTML = '';
            
            if (!alerts || alerts.length === 0) {
                recentAlertsContainer.innerHTML = '<div id="alerts-placeholder">No recent alerts</div>';
                // Also update alerts tab
                document.getElementById('all-alerts-container').innerHTML = '<div class="alert"><strong>No alerts to display</strong></div>';
                return;
            }
            
            // Show last 3 alerts on dashboard
            const recentAlerts = alerts.slice(0, 3);
            recentAlerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = alert.status === 'critical' ? 'alert alert-critical' : 'alert';
                
                alertDiv.innerHTML = `
                    <strong>${alert.worker_id}:</strong> ${alert.issues[0]}
                    <div class="timestamp">${formatDate(alert.timestamp)}</div>
                `;
                
                recentAlertsContainer.appendChild(alertDiv);
            });
            
            // Update alerts tab with all alerts
            const allAlertsContainer = document.getElementById('all-alerts-container');
            allAlertsContainer.innerHTML = '';
            
            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = alert.status === 'critical' ? 'alert alert-critical' : 'alert';
                
                alertDiv.innerHTML = `
                    <strong>${alert.worker_id} - ${alert.status}:</strong>
                    <ul>
                        ${alert.issues.map(issue => `<li>${issue}</li>`).join('')}
                    </ul>
                    <div class="timestamp">${formatDate(alert.timestamp)}</div>
                `;
                
                allAlertsContainer.appendChild(alertDiv);
            });
        }
        
        function updatePerformanceTables(workerPerformance, taskTypePerformance) {
            // Worker performance table
            const workerTable = document.getElementById('worker-performance-table');
            const workerTableBody = workerTable.querySelector('tbody');
            workerTableBody.innerHTML = '';
            
            if (!workerPerformance || Object.keys(workerPerformance).length === 0) {
                workerTableBody.innerHTML = '<tr><td colspan="5">No data available</td></tr>';
            } else {
                Object.entries(workerPerformance).forEach(([workerId, perf]) => {
                    const row = document.createElement('tr');
                    
                    let statusClass = '';
                    if (perf.status === 'healthy') statusClass = 'badge-healthy';
                    else if (perf.status === 'warning') statusClass = 'badge-warning';
                    else if (perf.status === 'critical') statusClass = 'badge-critical';
                    
                    row.innerHTML = `
                        <td>${workerId}</td>
                        <td>${perf.tasks_completed || 0}</td>
                        <td>${perf.success_rate ? (perf.success_rate * 100).toFixed(1) + '%' : 'N/A'}</td>
                        <td>${perf.avg_execution_time ? perf.avg_execution_time.toFixed(2) + 's' : 'N/A'}</td>
                        <td><span class="status-badge ${statusClass}">${perf.status || 'unknown'}</span></td>
                    `;
                    
                    workerTableBody.appendChild(row);
                });
            }
            
            // Task type performance table
            const taskTable = document.getElementById('task-type-performance-table');
            const taskTableBody = taskTable.querySelector('tbody');
            taskTableBody.innerHTML = '';
            
            if (!taskTypePerformance || Object.keys(taskTypePerformance).length === 0) {
                taskTableBody.innerHTML = '<tr><td colspan="5">No data available</td></tr>';
            } else {
                Object.entries(taskTypePerformance).forEach(([taskType, perf]) => {
                    const row = document.createElement('tr');
                    
                    row.innerHTML = `
                        <td>${taskType}</td>
                        <td>${perf.count || 0}</td>
                        <td>${perf.success_rate ? (perf.success_rate * 100).toFixed(1) + '%' : 'N/A'}</td>
                        <td>${perf.avg_execution_time ? perf.avg_execution_time.toFixed(2) + 's' : 'N/A'}</td>
                        <td>${perf.best_worker || 'N/A'}</td>
                    `;
                    
                    taskTableBody.appendChild(row);
                });
            }
        }
        
        function formatDate(dateString) {
            if (!dateString) return 'Unknown';
            
            try {
                const date = new Date(dateString);
                return date.toLocaleString();
            } catch (e) {
                return dateString;
            }
        }
    </script>
</body>
</html>
"""

class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""
    
    # Get coordinator URL from the server instance
    def __init__(self, *args, **kwargs):
        self.server_state = None
        super().__init__(*args, **kwargs)
        
    def do_GET(self):
        """Handle GET requests."""
        try:
            # Main dashboard page
            if self.path == '/' or self.path == '/index.html':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(HTML_TEMPLATE.encode())
                return
                
            # API for dashboard data
            elif self.path == '/api/dashboard':
                # Get dashboard data
                dashboard_data = self.server.dashboard_server.get_dashboard_data()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(dashboard_data).encode())
                return
                
            # Favicon or other static assets
            elif self.path == '/favicon.ico':
                self.send_response(204)  # No content
                self.end_headers()
                return
                
            # Not found
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'404 Not Found')
                return
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"Internal Server Error: {str(e)}".encode())


class DashboardServer:
    """Dashboard server for the distributed testing framework."""
    
    def __init__(self, host: str = "localhost", port: int = 8081,
                coordinator_url: str = None, auto_open: bool = False):
        """Initialize the dashboard server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            coordinator_url: URL of the coordinator server
            auto_open: Whether to automatically open the dashboard in a browser
        """
        self.host = host
        self.port = port
        self.coordinator_url = coordinator_url
        self.auto_open = auto_open
        self.server = None
        self.server_thread = None
        self.running = False
        
        # Data cache
        self.dashboard_data = {
            "workers": {
                "total": 0,
                "active": 0,
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "disconnected": 0
            },
            "tasks": {
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "avg_duration": 0
            },
            "worker_details": {},
            "recent_tasks": [],
            "queued_tasks": [],
            "running_tasks_details": [],
            "completed_tasks": [],
            "alerts": [],
            "worker_performance": {},
            "task_type_performance": {}
        }
        
        # Data update thread
        self.update_thread = None
        self.update_stop_event = threading.Event()
        self.update_interval = 5  # seconds
        
        # WebSocket client for real-time updates
        self.ws_client = None
        self.ws_thread = None
        self.ws_connected = False
        
        logger.info(f"Dashboard server initialized at {host}:{port}")
    
    def start(self):
        """Start the dashboard server."""
        # Create HTTPServer
        try:
            self.server = HTTPServer((self.host, self.port), DashboardHTTPHandler)
            self.server.dashboard_server = self  # Store reference to this instance
            
            # Create and start server thread
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            self.running = True
            
            logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
            
            # Start data update thread
            self.start_data_updates()
            
            # Try to connect to coordinator via WebSocket for real-time updates
            if self.coordinator_url and WEBSOCKET_AVAILABLE:
                self.connect_websocket()
                
            # Open dashboard in browser if requested
            if self.auto_open:
                self.open_in_browser()
                
            return True
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
            return False
    
    def stop(self):
        """Stop the dashboard server."""
        # Stop data update thread
        self.update_stop_event.set()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
            
        # Close WebSocket connection
        if self.ws_client:
            self.ws_client.close()
            
        # Stop HTTP server
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            
        self.running = False
        logger.info("Dashboard server stopped")
    
    def start_data_updates(self):
        """Start the data update thread."""
        if self.update_thread is not None and self.update_thread.is_alive():
            logger.warning("Data update thread already running")
            return
            
        self.update_stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        logger.info("Data update thread started")
    
    def _update_loop(self):
        """Data update thread function."""
        while not self.update_stop_event.is_set():
            try:
                # Fetch data from coordinator
                self._fetch_data_from_coordinator()
            except Exception as e:
                logger.error(f"Error updating dashboard data: {e}")
                
            # Wait for next update
            self.update_stop_event.wait(self.update_interval)
    
    def _fetch_data_from_coordinator(self):
        """Fetch dashboard data from the coordinator."""
        if not self.coordinator_url:
            logger.warning("No coordinator URL provided, cannot fetch data")
            return
            
        # Only fetch if not connected via WebSocket
        if self.ws_connected:
            return
            
        try:
            # Make HTTP request to coordinator API
            api_url = f"{self.coordinator_url.rstrip('/')}/api/dashboard"
            
            if REQUESTS_AVAILABLE:
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    self.dashboard_data = response.json()
            else:
                # Fallback to urllib
                with urllib.request.urlopen(api_url, timeout=5) as response:
                    self.dashboard_data = json.loads(response.read().decode())
                    
            logger.debug("Updated dashboard data from coordinator")
        except Exception as e:
            logger.error(f"Error fetching data from coordinator: {e}")
    
    def connect_websocket(self):
        """Connect to coordinator via WebSocket for real-time updates."""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("websocket-client not available, cannot connect via WebSocket")
            return
            
        if not self.coordinator_url:
            logger.warning("No coordinator URL provided, cannot connect via WebSocket")
            return
            
        # Convert HTTP URL to WebSocket URL
        ws_url = self.coordinator_url.replace('http://', 'ws://').replace('https://', 'wss://')
        ws_url = f"{ws_url.rstrip('/')}/ws/dashboard"
        
        try:
            # Create WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(
                target=self._websocket_thread,
                args=(ws_url,),
                daemon=True
            )
            self.ws_thread.start()
            logger.info(f"Started WebSocket connection thread to {ws_url}")
        except Exception as e:
            logger.error(f"Error creating WebSocket thread: {e}")
    
    def _websocket_thread(self, ws_url: str):
        """WebSocket client thread function.
        
        Args:
            ws_url: WebSocket URL to connect to
        """
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("type") == "dashboard_update":
                    self.dashboard_data = data.get("data", {})
                    logger.debug("Received dashboard update via WebSocket")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.ws_connected = False
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_msg}")
            self.ws_connected = False
            
            # Try to reconnect after delay
            time.sleep(5)
            if not self.update_stop_event.is_set():
                self.connect_websocket()
                
        def on_open(ws):
            logger.info("WebSocket connection established")
            self.ws_connected = True
        
        # Create and connect WebSocket client
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            self.ws_client = ws
            ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket thread error: {e}")
            self.ws_connected = False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data.
        
        Returns:
            Dict containing dashboard data
        """
        return self.dashboard_data
    
    def open_in_browser(self):
        """Open the dashboard in a web browser."""
        dashboard_url = f"http://{self.host}:{self.port}"
        
        try:
            # Check if host is 0.0.0.0, use localhost instead for browser
            if self.host == "0.0.0.0":
                dashboard_url = f"http://localhost:{self.port}"
                
            logger.info(f"Opening dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        except Exception as e:
            logger.error(f"Error opening dashboard in browser: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed Testing Framework Dashboard Server")
    
    parser.add_argument("--host", default="localhost",
                      help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8081,
                      help="Port to bind the server to")
    parser.add_argument("--coordinator-url", required=True,
                      help="URL of the coordinator server")
    parser.add_argument("--auto-open", action="store_true",
                      help="Automatically open dashboard in web browser")
    
    args = parser.parse_args()
    
    # Create dashboard server
    dashboard_server = DashboardServer(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator_url,
        auto_open=args.auto_open
    )
    
    # Start server
    try:
        logger.info("Starting dashboard server...")
        dashboard_server.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        dashboard_server.stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())