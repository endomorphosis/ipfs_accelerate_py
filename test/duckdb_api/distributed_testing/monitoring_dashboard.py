#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Monitoring Dashboard for the Distributed Testing Framework.

This module provides a web-based dashboard for monitoring, visualizing, and managing
the Distributed Testing Framework, including worker status, task execution, resource
utilization, error tracking, and performance metrics.
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import anyio
import traceback
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

import websockets
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DashboardMetrics:
    """Class for collecting and processing dashboard metrics."""
    
    def __init__(self, db_path=None):
        """
        Initialize the dashboard metrics processor.
        
        Args:
            db_path (str, optional): Path to SQLite database. If None, uses in-memory DB.
        """
        self.db_path = db_path or ":memory:"
        self.conn = self._init_database()
        self.metric_calculators = {}
        self.visualization_generators = {}
        
        # Initialize standard metrics
        self._init_standard_metrics()
        self._init_standard_visualizations()
        
        logger.info(f"Dashboard metrics initialized with database at {self.db_path}")
    
    def _init_database(self):
        """
        Initialize the metrics database.
        
        Returns:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY,
            metric_name TEXT,
            metric_value REAL,
            metric_type TEXT,
            category TEXT,
            entity_id TEXT,
            entity_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            event_type TEXT,
            event_data TEXT,
            entity_id TEXT,
            entity_type TEXT,
            severity TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            entity_id TEXT UNIQUE,
            entity_type TEXT,
            entity_name TEXT,
            entity_data TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            task_id TEXT UNIQUE,
            task_type TEXT,
            task_data TEXT,
            worker_id TEXT,
            status TEXT,
            priority INTEGER,
            execution_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            alert_type TEXT,
            alert_message TEXT,
            severity TEXT,
            entity_id TEXT,
            entity_type TEXT,
            is_active INTEGER,
            is_acknowledged INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            acknowledged_at TIMESTAMP
        )
        ''')
        
        conn.commit()
        return conn
    
    def _init_standard_metrics(self):
        """Initialize standard metric calculators."""
        # Worker metrics
        self.register_metric_calculator(
            "worker_count", 
            lambda: self._count_entities("worker")
        )
        
        self.register_metric_calculator(
            "active_worker_count", 
            lambda: self._count_entities("worker", status="active")
        )
        
        # Task metrics
        self.register_metric_calculator(
            "task_count", 
            lambda: self._count_entities("task")
        )
        
        self.register_metric_calculator(
            "pending_task_count", 
            lambda: self._count_tasks(status="pending")
        )
        
        self.register_metric_calculator(
            "running_task_count", 
            lambda: self._count_tasks(status="running")
        )
        
        self.register_metric_calculator(
            "completed_task_count", 
            lambda: self._count_tasks(status="completed")
        )
        
        self.register_metric_calculator(
            "failed_task_count", 
            lambda: self._count_tasks(status="failed")
        )
        
        # Performance metrics
        self.register_metric_calculator(
            "average_task_execution_time", 
            lambda: self._calculate_avg_task_execution_time()
        )
        
        self.register_metric_calculator(
            "task_throughput", 
            lambda: self._calculate_task_throughput()
        )
        
        # Error metrics
        self.register_metric_calculator(
            "error_count", 
            lambda: self._count_events(event_type="error")
        )
        
        self.register_metric_calculator(
            "active_alert_count", 
            lambda: self._count_alerts(is_active=1)
        )
    
    def _init_standard_visualizations(self):
        """Initialize standard visualization generators."""
        # Worker status visualization
        self.register_visualization_generator(
            "worker_status_chart",
            self._generate_worker_status_chart
        )
        
        # Task status visualization
        self.register_visualization_generator(
            "task_status_chart",
            self._generate_task_status_chart
        )
        
        # Task execution time visualization
        self.register_visualization_generator(
            "task_execution_time_chart",
            self._generate_task_execution_time_chart
        )
        
        # Task throughput visualization
        self.register_visualization_generator(
            "task_throughput_chart",
            self._generate_task_throughput_chart
        )
        
        # Error distribution visualization
        self.register_visualization_generator(
            "error_distribution_chart",
            self._generate_error_distribution_chart
        )
        
        # Resource utilization visualization
        self.register_visualization_generator(
            "resource_utilization_chart",
            self._generate_resource_utilization_chart
        )
    
    def register_metric_calculator(self, metric_name: str, calculator_func: Callable) -> None:
        """
        Register a calculator function for a metric.
        
        Args:
            metric_name: Name of the metric
            calculator_func: Function that calculates the metric
        """
        self.metric_calculators[metric_name] = calculator_func
        logger.info(f"Registered metric calculator for {metric_name}")
    
    def register_visualization_generator(self, viz_name: str, generator_func: Callable) -> None:
        """
        Register a generator function for a visualization.
        
        Args:
            viz_name: Name of the visualization
            generator_func: Function that generates the visualization
        """
        self.visualization_generators[viz_name] = generator_func
        logger.info(f"Registered visualization generator for {viz_name}")
    
    def record_metric(self, 
                      metric_name: str, 
                      metric_value: float, 
                      metric_type: str = "gauge",
                      category: str = "general",
                      entity_id: Optional[str] = None,
                      entity_type: Optional[str] = None) -> int:
        """
        Record a metric in the database.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric (gauge, counter, histogram)
            category: Category of the metric
            entity_id: ID of the entity associated with the metric
            entity_type: Type of the entity associated with the metric
            
        Returns:
            int: ID of the recorded metric
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO metrics 
        (metric_name, metric_value, metric_type, category, entity_id, entity_type, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (metric_name, metric_value, metric_type, category, entity_id, entity_type))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def record_event(self, 
                    event_type: str, 
                    event_data: Dict[str, Any],
                    entity_id: Optional[str] = None,
                    entity_type: Optional[str] = None,
                    severity: str = "info") -> int:
        """
        Record an event in the database.
        
        Args:
            event_type: Type of the event
            event_data: Data associated with the event
            entity_id: ID of the entity associated with the event
            entity_type: Type of the entity associated with the event
            severity: Severity of the event
            
        Returns:
            int: ID of the recorded event
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO events 
        (event_type, event_data, entity_id, entity_type, severity, timestamp)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ''', (event_type, json.dumps(event_data), entity_id, entity_type, severity))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def record_entity(self, 
                     entity_id: str, 
                     entity_type: str,
                     entity_name: str,
                     entity_data: Dict[str, Any],
                     status: str = "active") -> int:
        """
        Record or update an entity in the database.
        
        Args:
            entity_id: ID of the entity
            entity_type: Type of the entity
            entity_name: Name of the entity
            entity_data: Data associated with the entity
            status: Status of the entity
            
        Returns:
            int: ID of the recorded entity
        """
        cursor = self.conn.cursor()
        
        # Check if entity exists
        cursor.execute('''
        SELECT id FROM entities WHERE entity_id = ?
        ''', (entity_id,))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing entity
            cursor.execute('''
            UPDATE entities 
            SET entity_type = ?, entity_name = ?, entity_data = ?, status = ?, updated_at = datetime('now')
            WHERE entity_id = ?
            ''', (entity_type, entity_name, json.dumps(entity_data), status, entity_id))
            entity_id = existing[0]
        else:
            # Insert new entity
            cursor.execute('''
            INSERT INTO entities 
            (entity_id, entity_type, entity_name, entity_data, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''', (entity_id, entity_type, entity_name, json.dumps(entity_data), status))
            entity_id = cursor.lastrowid
        
        self.conn.commit()
        return entity_id
    
    def record_task(self, 
                   task_id: str, 
                   task_type: str,
                   task_data: Dict[str, Any],
                   status: str = "pending",
                   worker_id: Optional[str] = None,
                   priority: int = 5) -> int:
        """
        Record or update a task in the database.
        
        Args:
            task_id: ID of the task
            task_type: Type of the task
            task_data: Data associated with the task
            status: Status of the task
            worker_id: ID of the worker assigned to the task
            priority: Priority of the task (1-10, lower is higher priority)
            
        Returns:
            int: ID of the recorded task
        """
        cursor = self.conn.cursor()
        
        # Check if task exists
        cursor.execute('''
        SELECT id FROM tasks WHERE task_id = ?
        ''', (task_id,))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing task
            cursor.execute('''
            UPDATE tasks 
            SET task_type = ?, task_data = ?, status = ?, worker_id = ?, priority = ?, updated_at = datetime('now')
            WHERE task_id = ?
            ''', (task_type, json.dumps(task_data), status, worker_id, priority, task_id))
            task_id = existing[0]
            
            # If task completed, record completion time and execution time
            if status == 'completed' or status == 'failed':
                # Get created_at time
                cursor.execute('''
                SELECT created_at FROM tasks WHERE id = ?
                ''', (task_id,))
                created_at = cursor.fetchone()[0]
                
                # Calculate execution time
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                completed_dt = datetime.now()
                execution_time = (completed_dt - created_dt).total_seconds()
                
                # Update completion data
                cursor.execute('''
                UPDATE tasks 
                SET completed_at = datetime('now'), execution_time = ?
                WHERE id = ?
                ''', (execution_time, task_id))
                
                # Record execution time metric
                self.record_metric(
                    metric_name="task_execution_time",
                    metric_value=execution_time,
                    metric_type="gauge",
                    category="performance",
                    entity_id=task_id,
                    entity_type="task"
                )
        else:
            # Insert new task
            cursor.execute('''
            INSERT INTO tasks 
            (task_id, task_type, task_data, status, worker_id, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''', (task_id, task_type, json.dumps(task_data), status, worker_id, priority))
            task_id = cursor.lastrowid
        
        self.conn.commit()
        return task_id
    
    def record_alert(self, 
                    alert_type: str, 
                    alert_message: str,
                    severity: str = "warning",
                    entity_id: Optional[str] = None,
                    entity_type: Optional[str] = None) -> int:
        """
        Record an alert in the database.
        
        Args:
            alert_type: Type of the alert
            alert_message: Message for the alert
            severity: Severity of the alert
            entity_id: ID of the entity associated with the alert
            entity_type: Type of the entity associated with the alert
            
        Returns:
            int: ID of the recorded alert
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO alerts 
        (alert_type, alert_message, severity, entity_id, entity_type, is_active, is_acknowledged, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, 0, datetime('now'), datetime('now'))
        ''', (alert_type, alert_message, severity, entity_id, entity_type))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            bool: True if acknowledged, False otherwise
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        UPDATE alerts 
        SET is_acknowledged = 1, acknowledged_at = datetime('now'), updated_at = datetime('now')
        WHERE id = ?
        ''', (alert_id,))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def resolve_alert(self, alert_id: int) -> bool:
        """
        Resolve (deactivate) an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            bool: True if resolved, False otherwise
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        UPDATE alerts 
        SET is_active = 0, updated_at = datetime('now')
        WHERE id = ?
        ''', (alert_id,))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def get_metric(self, 
                  metric_name: str, 
                  entity_id: Optional[str] = None,
                  time_window: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get metrics from the database.
        
        Args:
            metric_name: Name of the metric to get
            entity_id: Filter by entity ID
            time_window: Time window in seconds
            
        Returns:
            List[Dict]: List of metric records
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM metrics WHERE metric_name = ?"
        params = [metric_name]
        
        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)
        
        if time_window:
            query += " AND timestamp > datetime('now', '-' || ? || ' seconds')"
            params.append(time_window)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_events(self, 
                  event_type: Optional[str] = None, 
                  entity_id: Optional[str] = None,
                  time_window: Optional[int] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get events from the database.
        
        Args:
            event_type: Filter by event type
            entity_id: Filter by entity ID
            time_window: Time window in seconds
            limit: Maximum number of events to return
            
        Returns:
            List[Dict]: List of event records
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM events"
        params = []
        
        conditions = []
        
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        
        if entity_id:
            conditions.append("entity_id = ?")
            params.append(entity_id)
        
        if time_window:
            conditions.append("timestamp > datetime('now', '-' || ? || ' seconds')")
            params.append(time_window)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        events = []
        for row in cursor.fetchall():
            event = dict(zip(columns, row))
            # Parse event data JSON
            if event['event_data']:
                event['event_data'] = json.loads(event['event_data'])
            events.append(event)
        
        return events
    
    def get_entities(self, 
                    entity_type: Optional[str] = None, 
                    status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get entities from the database.
        
        Args:
            entity_type: Filter by entity type
            status: Filter by status
            
        Returns:
            List[Dict]: List of entity records
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM entities"
        params = []
        
        conditions = []
        
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY updated_at DESC"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        entities = []
        for row in cursor.fetchall():
            entity = dict(zip(columns, row))
            # Parse entity data JSON
            if entity['entity_data']:
                entity['entity_data'] = json.loads(entity['entity_data'])
            entities.append(entity)
        
        return entities
    
    def get_tasks(self, 
                 task_type: Optional[str] = None, 
                 status: Optional[str] = None,
                 worker_id: Optional[str] = None,
                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get tasks from the database.
        
        Args:
            task_type: Filter by task type
            status: Filter by status
            worker_id: Filter by worker ID
            limit: Maximum number of tasks to return
            
        Returns:
            List[Dict]: List of task records
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM tasks"
        params = []
        
        conditions = []
        
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if worker_id:
            conditions.append("worker_id = ?")
            params.append(worker_id)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        tasks = []
        for row in cursor.fetchall():
            task = dict(zip(columns, row))
            # Parse task data JSON
            if task['task_data']:
                task['task_data'] = json.loads(task['task_data'])
            tasks.append(task)
        
        return tasks
    
    def get_alerts(self, 
                  alert_type: Optional[str] = None, 
                  is_active: Optional[int] = None,
                  is_acknowledged: Optional[int] = None,
                  entity_id: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alerts from the database.
        
        Args:
            alert_type: Filter by alert type
            is_active: Filter by active status
            is_acknowledged: Filter by acknowledged status
            entity_id: Filter by entity ID
            limit: Maximum number of alerts to return
            
        Returns:
            List[Dict]: List of alert records
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM alerts"
        params = []
        
        conditions = []
        
        if alert_type:
            conditions.append("alert_type = ?")
            params.append(alert_type)
        
        if is_active is not None:
            conditions.append("is_active = ?")
            params.append(is_active)
        
        if is_acknowledged is not None:
            conditions.append("is_acknowledged = ?")
            params.append(is_acknowledged)
        
        if entity_id:
            conditions.append("entity_id = ?")
            params.append(entity_id)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all registered metrics.
        
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        metrics = {}
        
        for metric_name, calculator_func in self.metric_calculators.items():
            try:
                metrics[metric_name] = calculator_func()
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {e}")
                metrics[metric_name] = None
        
        return metrics
    
    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Generate all registered visualizations.
        
        Returns:
            Dict[str, Any]: Dictionary of visualization names and data
        """
        visualizations = {}
        
        for viz_name, generator_func in self.visualization_generators.items():
            try:
                visualizations[viz_name] = generator_func()
            except Exception as e:
                logger.error(f"Error generating visualization {viz_name}: {e}")
                visualizations[viz_name] = None
        
        return visualizations
    
    def _count_entities(self, entity_type: str, status: Optional[str] = None) -> int:
        """
        Count entities of a specific type.
        
        Args:
            entity_type: Type of entity to count
            status: Filter by status
            
        Returns:
            int: Count of entities
        """
        cursor = self.conn.cursor()
        
        query = "SELECT COUNT(*) FROM entities WHERE entity_type = ?"
        params = [entity_type]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]
    
    def _count_tasks(self, status: Optional[str] = None) -> int:
        """
        Count tasks.
        
        Args:
            status: Filter by status
            
        Returns:
            int: Count of tasks
        """
        cursor = self.conn.cursor()
        
        query = "SELECT COUNT(*) FROM tasks"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]
    
    def _count_events(self, event_type: Optional[str] = None) -> int:
        """
        Count events.
        
        Args:
            event_type: Filter by event type
            
        Returns:
            int: Count of events
        """
        cursor = self.conn.cursor()
        
        query = "SELECT COUNT(*) FROM events"
        params = []
        
        if event_type:
            query += " WHERE event_type = ?"
            params.append(event_type)
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]
    
    def _count_alerts(self, is_active: Optional[int] = None) -> int:
        """
        Count alerts.
        
        Args:
            is_active: Filter by active status
            
        Returns:
            int: Count of alerts
        """
        cursor = self.conn.cursor()
        
        query = "SELECT COUNT(*) FROM alerts"
        params = []
        
        if is_active is not None:
            query += " WHERE is_active = ?"
            params.append(is_active)
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]
    
    def _calculate_avg_task_execution_time(self) -> float:
        """
        Calculate average task execution time.
        
        Returns:
            float: Average execution time
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT AVG(execution_time) 
        FROM tasks 
        WHERE execution_time IS NOT NULL AND status = 'completed'
        ''')
        
        result = cursor.fetchone()[0]
        return float(result) if result else 0.0
    
    def _calculate_task_throughput(self) -> float:
        """
        Calculate task throughput (tasks completed per minute).
        
        Returns:
            float: Tasks completed per minute
        """
        cursor = self.conn.cursor()
        
        # Count tasks completed in the last hour
        cursor.execute('''
        SELECT COUNT(*) 
        FROM tasks 
        WHERE status = 'completed' 
        AND completed_at > datetime('now', '-1 hour')
        ''')
        
        completed_count = cursor.fetchone()[0]
        
        # If no tasks completed, return 0
        if completed_count == 0:
            return 0.0
        
        # Calculate throughput (tasks per minute)
        return completed_count / 60.0
    
    def _generate_worker_status_chart(self) -> Dict[str, Any]:
        """
        Generate worker status chart data.
        
        Returns:
            Dict: Chart data
        """
        cursor = self.conn.cursor()
        
        # Get worker status counts
        cursor.execute('''
        SELECT status, COUNT(*) 
        FROM entities 
        WHERE entity_type = 'worker' 
        GROUP BY status
        ''')
        
        status_counts = dict(cursor.fetchall())
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=.3,
            marker_colors=['#00cc96', '#ef553b', '#636efa', '#ffa15a']
        )])
        
        fig.update_layout(
            title_text='Worker Status Distribution',
            template='plotly_white'
        )
        
        # Convert to JSON
        chart_json = pio.to_json(fig)
        
        return {
            "type": "plotly",
            "data": chart_json
        }
    
    def _generate_task_status_chart(self) -> Dict[str, Any]:
        """
        Generate task status chart data.
        
        Returns:
            Dict: Chart data
        """
        cursor = self.conn.cursor()
        
        # Get task status counts
        cursor.execute('''
        SELECT status, COUNT(*) 
        FROM tasks 
        GROUP BY status
        ''')
        
        status_counts = dict(cursor.fetchall())
        
        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=list(status_counts.keys()),
            y=list(status_counts.values()),
            marker_color=['#ffa15a', '#00cc96', '#ef553b', '#636efa']
        )])
        
        fig.update_layout(
            title_text='Task Status Distribution',
            xaxis_title='Status',
            yaxis_title='Count',
            template='plotly_white'
        )
        
        # Convert to JSON
        chart_json = pio.to_json(fig)
        
        return {
            "type": "plotly",
            "data": chart_json
        }
    
    def _generate_task_execution_time_chart(self) -> Dict[str, Any]:
        """
        Generate task execution time chart data.
        
        Returns:
            Dict: Chart data
        """
        cursor = self.conn.cursor()
        
        # Get recent task execution times
        cursor.execute('''
        SELECT task_id, task_type, execution_time, completed_at
        FROM tasks 
        WHERE execution_time IS NOT NULL AND status = 'completed'
        ORDER BY completed_at DESC
        LIMIT 100
        ''')
        
        task_data = cursor.fetchall()
        
        if not task_data:
            return {"type": "none", "message": "No task execution data available"}
        
        # Create DataFrame
        df = pd.DataFrame(task_data, columns=['task_id', 'task_type', 'execution_time', 'completed_at'])
        
        # Group by task type
        task_type_avg = df.groupby('task_type')['execution_time'].mean().reset_index()
        
        # Create bar chart for average execution time by task type
        fig1 = go.Figure(data=[go.Bar(
            x=task_type_avg['task_type'],
            y=task_type_avg['execution_time'],
            marker_color='#636efa'
        )])
        
        fig1.update_layout(
            title_text='Average Execution Time by Task Type',
            xaxis_title='Task Type',
            yaxis_title='Execution Time (s)',
            template='plotly_white'
        )
        
        # Create line chart for execution time trend
        fig2 = go.Figure(data=[go.Scatter(
            x=df['completed_at'],
            y=df['execution_time'],
            mode='lines+markers',
            marker_color='#00cc96'
        )])
        
        fig2.update_layout(
            title_text='Task Execution Time Trend',
            xaxis_title='Completion Time',
            yaxis_title='Execution Time (s)',
            template='plotly_white'
        )
        
        # Convert to JSON
        chart1_json = pio.to_json(fig1)
        chart2_json = pio.to_json(fig2)
        
        return {
            "type": "plotly_multi",
            "charts": [
                {"name": "execution_time_by_type", "data": chart1_json},
                {"name": "execution_time_trend", "data": chart2_json}
            ]
        }
    
    def _generate_task_throughput_chart(self) -> Dict[str, Any]:
        """
        Generate task throughput chart data.
        
        Returns:
            Dict: Chart data
        """
        cursor = self.conn.cursor()
        
        # Get task completion data by hour
        cursor.execute('''
        SELECT strftime('%Y-%m-%d %H:00:00', completed_at) as hour, COUNT(*) as count
        FROM tasks 
        WHERE status = 'completed' AND completed_at IS NOT NULL
        GROUP BY strftime('%Y-%m-%d %H:00:00', completed_at)
        ORDER BY hour
        LIMIT 24
        ''')
        
        throughput_data = cursor.fetchall()
        
        if not throughput_data:
            return {"type": "none", "message": "No task throughput data available"}
        
        # Create DataFrame
        df = pd.DataFrame(throughput_data, columns=['hour', 'count'])
        
        # Create line chart
        fig = go.Figure(data=[go.Scatter(
            x=df['hour'],
            y=df['count'],
            mode='lines+markers',
            marker_color='#00cc96'
        )])
        
        fig.update_layout(
            title_text='Task Throughput by Hour',
            xaxis_title='Hour',
            yaxis_title='Completed Tasks',
            template='plotly_white'
        )
        
        # Convert to JSON
        chart_json = pio.to_json(fig)
        
        return {
            "type": "plotly",
            "data": chart_json
        }
    
    def _generate_error_distribution_chart(self) -> Dict[str, Any]:
        """
        Generate error distribution chart data.
        
        Returns:
            Dict: Chart data
        """
        cursor = self.conn.cursor()
        
        # Get error events by type
        cursor.execute('''
        SELECT event_type, COUNT(*) as count
        FROM events 
        WHERE event_type LIKE '%error%' OR event_type LIKE '%exception%' OR event_type = 'error'
        GROUP BY event_type
        ORDER BY count DESC
        LIMIT 10
        ''')
        
        error_data = cursor.fetchall()
        
        if not error_data:
            return {"type": "none", "message": "No error data available"}
        
        # Create DataFrame
        df = pd.DataFrame(error_data, columns=['error_type', 'count'])
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=df['error_type'],
            values=df['count'],
            hole=.3,
            marker_colors=['#ef553b', '#ffa15a', '#ab63fa', '#fe6692']
        )])
        
        fig.update_layout(
            title_text='Error Distribution by Type',
            template='plotly_white'
        )
        
        # Convert to JSON
        chart_json = pio.to_json(fig)
        
        return {
            "type": "plotly",
            "data": chart_json
        }
    
    def _generate_resource_utilization_chart(self) -> Dict[str, Any]:
        """
        Generate resource utilization chart data.
        
        Returns:
            Dict: Chart data
        """
        # Get recent CPU, memory, and GPU utilization metrics
        cpu_metrics = self.get_metric('cpu_utilization', time_window=3600)
        memory_metrics = self.get_metric('memory_utilization', time_window=3600)
        gpu_metrics = self.get_metric('gpu_utilization', time_window=3600)
        
        if not cpu_metrics and not memory_metrics and not gpu_metrics:
            return {"type": "none", "message": "No resource utilization data available"}
        
        # Create subplots
        fig = make_subplots(rows=3, cols=1, 
                          subplot_titles=('CPU Utilization', 'Memory Utilization', 'GPU Utilization'),
                          vertical_spacing=0.1)
        
        # Add CPU utilization trace
        if cpu_metrics:
            df_cpu = pd.DataFrame(cpu_metrics)
            fig.add_trace(go.Scatter(
                x=df_cpu['timestamp'],
                y=df_cpu['metric_value'],
                mode='lines',
                name='CPU',
                line=dict(color='#636efa')
            ), row=1, col=1)
        
        # Add memory utilization trace
        if memory_metrics:
            df_memory = pd.DataFrame(memory_metrics)
            fig.add_trace(go.Scatter(
                x=df_memory['timestamp'],
                y=df_memory['metric_value'],
                mode='lines',
                name='Memory',
                line=dict(color='#ef553b')
            ), row=2, col=1)
        
        # Add GPU utilization trace
        if gpu_metrics:
            df_gpu = pd.DataFrame(gpu_metrics)
            fig.add_trace(go.Scatter(
                x=df_gpu['timestamp'],
                y=df_gpu['metric_value'],
                mode='lines',
                name='GPU',
                line=dict(color='#00cc96')
            ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text='Resource Utilization',
            template='plotly_white'
        )
        
        # Update y-axes
        fig.update_yaxes(range=[0, 100], ticksuffix='%')
        
        # Convert to JSON
        chart_json = pio.to_json(fig)
        
        return {
            "type": "plotly",
            "data": chart_json
        }


class MonitoringDashboard:
    """Monitoring dashboard server for the Distributed Testing Framework."""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 8080,
                 coordinator_url: Optional[str] = None,
                 db_path: Optional[str] = None,
                 auto_open: bool = False):
        """
        Initialize the monitoring dashboard.
        
        Args:
            host: Host to bind the dashboard server
            port: Port to bind the dashboard server
            coordinator_url: URL of the coordinator server
            db_path: Path to SQLite database for metrics
            auto_open: Whether to automatically open the dashboard in a browser
        """
        self.host = host
        self.port = port
        self.coordinator_url = coordinator_url
        self.auto_open = auto_open
        
        # Initialize dashboard metrics
        self.metrics = DashboardMetrics(db_path=db_path)
        
        # Connection to coordinator
        self.coordinator_connected = False
        self.coordinator_websocket = None
        
        # Client connections
        self.clients = set()
        
        # Initialize application
        self.app = self._create_app()
        
        # Background tasks
        self.stop_event = threading.Event()
        self.update_thread = None
        
        logger.info(f"Monitoring Dashboard initialized at http://{host}:{port}")
    
    def _create_app(self):
        """
        Create the Tornado web application.
        
        Returns:
            tornado.web.Application: Web application
        """
        static_path = os.path.join(os.path.dirname(__file__), "static")
        template_path = os.path.join(os.path.dirname(__file__), "templates")
        
        # Ensure static directory exists
        if not os.path.exists(static_path):
            os.makedirs(static_path)
            logger.info(f"Created static directory at {static_path}")
            
            # Create subdirectories
            for subdir in ["css", "js", "img"]:
                subdir_path = os.path.join(static_path, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
                    logger.info(f"Created static subdirectory at {subdir_path}")
        
        return tornado.web.Application([
            (r"/", MainHandler, {"dashboard": self}),
            (r"/ws", DashboardWebSocketHandler, {"dashboard": self}),
            (r"/api/metrics", MetricsHandler, {"dashboard": self}),
            (r"/api/events", EventsHandler, {"dashboard": self}),
            (r"/api/tasks", TasksHandler, {"dashboard": self}),
            (r"/api/workers", WorkersHandler, {"dashboard": self}),
            (r"/api/alerts", AlertsHandler, {"dashboard": self}),
            (r"/api/visualizations", VisualizationsHandler, {"dashboard": self}),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path}),
        ], 
        template_path=template_path,
        static_path=static_path,
        debug=True)
    
    def start(self):
        """Start the dashboard server."""
        # Start background threads
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Connect to coordinator
        if self.coordinator_url:
            threading.Thread(target=self._connect_to_coordinator).start()
        
        # Start Tornado server
        self.app.listen(self.port, address=self.host)
        
        # Open dashboard in browser
        if self.auto_open:
            import webbrowser
            webbrowser.open(f"http://{self.host}:{self.port}")
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
        
        # Start the IO loop
        tornado.ioloop.IOLoop.current().start()
    
    def stop(self):
        """Stop the dashboard server."""
        logger.info("Stopping dashboard server")
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        tornado.ioloop.IOLoop.current().stop()
    
    def register_client(self, client):
        """
        Register a client WebSocket connection.
        
        Args:
            client: WebSocket client
        """
        self.clients.add(client)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
    
    def unregister_client(self, client):
        """
        Unregister a client WebSocket connection.
        
        Args:
            client: WebSocket client
        """
        if client in self.clients:
            self.clients.remove(client)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    def broadcast_to_clients(self, message):
        """
        Broadcast a message to all clients.
        
        Args:
            message: Message to broadcast (must be JSON serializable)
        """
        if not self.clients:
            return
        
        message_json = json.dumps(message)
        
        for client in self.clients:
            try:
                tornado.ioloop.IOLoop.current().add_callback(
                    client.write_message, message_json
                )
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
    
    async def _connect_to_coordinator_async(self):
        """Connect to coordinator server asynchronously."""
        websocket_url = self.coordinator_url
        if not websocket_url.startswith("ws"):
            websocket_url = websocket_url.replace("http", "ws")
        
        logger.info(f"Connecting to coordinator at {websocket_url}")
        
        try:
            self.coordinator_websocket = await websockets.connect(websocket_url)
            self.coordinator_connected = True
            
            # Register dashboard
            register_msg = {
                "type": "register_dashboard",
                "dashboard_id": str(uuid.uuid4()),
                "name": "Distributed Testing Dashboard",
                "host": self.host,
                "port": self.port
            }
            await self.coordinator_websocket.send(json.dumps(register_msg))
            
            # Enter message loop
            while not self.stop_event.is_set():
                try:
                    message = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(
                        self.coordinator_websocket.recv(), timeout=1.0
                    )
                    await self._process_coordinator_message(message)
                except asyncio.TimeoutError:
                    # This is expected, just retry
                    continue
                except websockets.ConnectionClosed:
                    logger.error("Connection to coordinator closed")
                    self.coordinator_connected = False
                    break
                except Exception as e:
                    logger.error(f"Error processing coordinator message: {e}")
            
            # Close connection
            if self.coordinator_websocket and not self.coordinator_websocket.closed:
                await self.coordinator_websocket.close()
        
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            self.coordinator_connected = False
            self.coordinator_websocket = None
    
    def _connect_to_coordinator(self):
        """Connect to coordinator server in a background thread."""
        # Create a new event loop for the thread
        loop = # TODO: Remove event loop management - asyncio.new_event_loop()
        # TODO: Remove event loop management - asyncio.set_event_loop(loop)
        
        while not self.stop_event.is_set():
            # Try to connect
            loop.run_until_complete(self._connect_to_coordinator_async())
            
            # If connection failed or closed, wait before retrying
            if not self.coordinator_connected and not self.stop_event.is_set():
                logger.info("Waiting 5 seconds before reconnecting to coordinator")
                time.sleep(5)
            else:
                # If stopped or successfully connected and then disconnected, exit loop
                break
    
    async def _process_coordinator_message(self, message):
        """
        Process a message from the coordinator.
        
        Args:
            message: JSON message from coordinator
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "worker_update":
                # Update worker information
                worker_id = data.get("worker_id")
                worker_info = data.get("worker_info", {})
                worker_status = data.get("status", "active")
                
                self.metrics.record_entity(
                    entity_id=worker_id,
                    entity_type="worker",
                    entity_name=worker_info.get("hostname", worker_id),
                    entity_data=worker_info,
                    status=worker_status
                )
            
            elif message_type == "task_update":
                # Update task information
                task_id = data.get("task_id")
                task_info = data.get("task_info", {})
                task_status = data.get("status", "pending")
                worker_id = data.get("worker_id")
                
                self.metrics.record_task(
                    task_id=task_id,
                    task_type=task_info.get("type", "unknown"),
                    task_data=task_info,
                    status=task_status,
                    worker_id=worker_id
                )
            
            elif message_type == "error":
                # Record error event
                error_info = data.get("error_info", {})
                entity_id = data.get("entity_id")
                entity_type = data.get("entity_type")
                
                self.metrics.record_event(
                    event_type="error",
                    event_data=error_info,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    severity="error"
                )
                
                # Also create an alert
                self.metrics.record_alert(
                    alert_type="error",
                    alert_message=error_info.get("message", "Unknown error"),
                    severity="warning",
                    entity_id=entity_id,
                    entity_type=entity_type
                )
            
            elif message_type == "resource_usage":
                # Record resource usage metrics
                worker_id = data.get("worker_id")
                resources = data.get("resources", {})
                
                # Record CPU utilization
                if "cpu" in resources:
                    cpu_usage = resources["cpu"].get("usage_percent", 0)
                    self.metrics.record_metric(
                        metric_name="cpu_utilization",
                        metric_value=cpu_usage,
                        entity_id=worker_id,
                        entity_type="worker",
                        category="resources"
                    )
                
                # Record memory utilization
                if "memory" in resources:
                    memory_usage = resources["memory"].get("usage_percent", 0)
                    self.metrics.record_metric(
                        metric_name="memory_utilization",
                        metric_value=memory_usage,
                        entity_id=worker_id,
                        entity_type="worker",
                        category="resources"
                    )
                
                # Record GPU utilization
                if "gpu" in resources:
                    gpu_usage = resources["gpu"].get("usage_percent", 0)
                    self.metrics.record_metric(
                        metric_name="gpu_utilization",
                        metric_value=gpu_usage,
                        entity_id=worker_id,
                        entity_type="worker",
                        category="resources"
                    )
            
            # Broadcast message to clients
            self.broadcast_to_clients({
                "type": "coordinator_message",
                "message": data
            })
            
        except Exception as e:
            logger.error(f"Error processing coordinator message: {e}")
            logger.error(traceback.format_exc())
    
    def _update_loop(self):
        """Background thread for periodic updates."""
        while not self.stop_event.is_set():
            try:
                # Calculate metrics
                metrics = self.metrics.calculate_metrics()
                
                # Broadcast metrics to clients
                self.broadcast_to_clients({
                    "type": "metrics_update",
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check for pending alerts
                pending_alerts = self.metrics.get_alerts(is_active=1, is_acknowledged=0)
                
                if pending_alerts:
                    # Broadcast alerts to clients
                    self.broadcast_to_clients({
                        "type": "alerts_update",
                        "alerts": pending_alerts,
                        "count": len(pending_alerts),
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Sleep for 5 seconds
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(10)  # Sleep longer on error


class MainHandler(tornado.web.RequestHandler):
    """Main request handler for the dashboard."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request."""
        self.render("index.html", 
                   title="Distributed Testing Dashboard",
                   host=self.dashboard.host,
                   port=self.dashboard.port,
                   coordinator_url=self.dashboard.coordinator_url,
                   coordinator_connected=self.dashboard.coordinator_connected)


class DashboardWebSocketHandler(tornado.websocket.WebSocketHandler):
    """WebSocket handler for dashboard client connections."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def check_origin(self, origin):
        """Allow connections from any origin."""
        return True
    
    def open(self):
        """Handle WebSocket connection open."""
        self.dashboard.register_client(self)
    
    def on_message(self, message):
        """
        Handle incoming WebSocket message.
        
        Args:
            message: Client message
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "get_metrics":
                # Send current metrics to client
                metrics = self.dashboard.metrics.calculate_metrics()
                self.write_message(json.dumps({
                    "type": "metrics_update",
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message_type == "get_workers":
                # Send worker information to client
                workers = self.dashboard.metrics.get_entities(entity_type="worker")
                self.write_message(json.dumps({
                    "type": "workers_update",
                    "workers": workers,
                    "count": len(workers),
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message_type == "get_tasks":
                # Send task information to client
                tasks = self.dashboard.metrics.get_tasks()
                self.write_message(json.dumps({
                    "type": "tasks_update",
                    "tasks": tasks,
                    "count": len(tasks),
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message_type == "get_alerts":
                # Send alert information to client
                alerts = self.dashboard.metrics.get_alerts(is_active=1)
                self.write_message(json.dumps({
                    "type": "alerts_update",
                    "alerts": alerts,
                    "count": len(alerts),
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message_type == "get_visualization":
                # Send visualization data to client
                viz_name = data.get("visualization")
                visualizations = self.dashboard.metrics.generate_visualizations()
                
                if viz_name and viz_name in visualizations:
                    viz_data = visualizations[viz_name]
                else:
                    viz_data = visualizations
                
                self.write_message(json.dumps({
                    "type": "visualization_update",
                    "visualization": viz_name if viz_name else "all",
                    "data": viz_data,
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message_type == "acknowledge_alert":
                # Acknowledge an alert
                alert_id = data.get("alert_id")
                if alert_id:
                    self.dashboard.metrics.acknowledge_alert(int(alert_id))
                    self.write_message(json.dumps({
                        "type": "alert_acknowledged",
                        "alert_id": alert_id,
                        "success": True
                    }))
            
            elif message_type == "resolve_alert":
                # Resolve an alert
                alert_id = data.get("alert_id")
                if alert_id:
                    self.dashboard.metrics.resolve_alert(int(alert_id))
                    self.write_message(json.dumps({
                        "type": "alert_resolved",
                        "alert_id": alert_id,
                        "success": True
                    }))
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            self.write_message(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    def on_close(self):
        """Handle WebSocket connection close."""
        self.dashboard.unregister_client(self)


class MetricsHandler(tornado.web.RequestHandler):
    """REST API handler for metrics."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for metrics."""
        metric_name = self.get_argument("metric", None)
        entity_id = self.get_argument("entity_id", None)
        time_window = self.get_argument("time_window", None)
        
        if time_window:
            time_window = int(time_window)
        
        if metric_name:
            metrics = self.dashboard.metrics.get_metric(
                metric_name=metric_name,
                entity_id=entity_id,
                time_window=time_window
            )
        else:
            metrics = self.dashboard.metrics.calculate_metrics()
        
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(metrics))


class EventsHandler(tornado.web.RequestHandler):
    """REST API handler for events."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for events."""
        event_type = self.get_argument("event_type", None)
        entity_id = self.get_argument("entity_id", None)
        time_window = self.get_argument("time_window", None)
        limit = self.get_argument("limit", 100)
        
        if time_window:
            time_window = int(time_window)
        
        events = self.dashboard.metrics.get_events(
            event_type=event_type,
            entity_id=entity_id,
            time_window=time_window,
            limit=int(limit)
        )
        
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(events))


class TasksHandler(tornado.web.RequestHandler):
    """REST API handler for tasks."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for tasks."""
        task_type = self.get_argument("task_type", None)
        status = self.get_argument("status", None)
        worker_id = self.get_argument("worker_id", None)
        limit = self.get_argument("limit", 100)
        
        tasks = self.dashboard.metrics.get_tasks(
            task_type=task_type,
            status=status,
            worker_id=worker_id,
            limit=int(limit)
        )
        
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(tasks))


class WorkersHandler(tornado.web.RequestHandler):
    """REST API handler for workers."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for workers."""
        status = self.get_argument("status", None)
        
        workers = self.dashboard.metrics.get_entities(
            entity_type="worker",
            status=status
        )
        
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(workers))


class AlertsHandler(tornado.web.RequestHandler):
    """REST API handler for alerts."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for alerts."""
        alert_type = self.get_argument("alert_type", None)
        is_active = self.get_argument("is_active", None)
        is_acknowledged = self.get_argument("is_acknowledged", None)
        entity_id = self.get_argument("entity_id", None)
        limit = self.get_argument("limit", 100)
        
        if is_active is not None:
            is_active = int(is_active)
        if is_acknowledged is not None:
            is_acknowledged = int(is_acknowledged)
        
        alerts = self.dashboard.metrics.get_alerts(
            alert_type=alert_type,
            is_active=is_active,
            is_acknowledged=is_acknowledged,
            entity_id=entity_id,
            limit=int(limit)
        )
        
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(alerts))
    
    def post(self):
        """Handle POST request for alerts (acknowledgement or resolution)."""
        try:
            data = json.loads(self.request.body)
            alert_id = data.get("alert_id")
            action = data.get("action")
            
            if not alert_id or not action:
                self.set_status(400)
                self.write({"error": "Missing required fields"})
                return
            
            if action == "acknowledge":
                success = self.dashboard.metrics.acknowledge_alert(int(alert_id))
            elif action == "resolve":
                success = self.dashboard.metrics.resolve_alert(int(alert_id))
            else:
                self.set_status(400)
                self.write({"error": f"Invalid action: {action}"})
                return
            
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"success": success}))
            
        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})


class VisualizationsHandler(tornado.web.RequestHandler):
    """REST API handler for visualizations."""
    
    def initialize(self, dashboard):
        """
        Initialize the handler.
        
        Args:
            dashboard: Dashboard instance
        """
        self.dashboard = dashboard
    
    def get(self):
        """Handle GET request for visualizations."""
        viz_name = self.get_argument("name", None)
        
        visualizations = self.dashboard.metrics.generate_visualizations()
        
        if viz_name and viz_name in visualizations:
            visualization = {viz_name: visualizations[viz_name]}
        else:
            visualization = visualizations
        
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(visualization))


def main():
    """Main entry point for the dashboard server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitoring Dashboard for Distributed Testing Framework")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the dashboard server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the dashboard server")
    parser.add_argument("--coordinator-url", type=str, help="URL of the coordinator server")
    parser.add_argument("--db-path", type=str, help="Path to SQLite database for metrics")
    parser.add_argument("--auto-open", action="store_true", help="Automatically open the dashboard in a browser")
    
    args = parser.parse_args()
    
    dashboard = MonitoringDashboard(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator_url,
        db_path=args.db_path,
        auto_open=args.auto_open
    )
    
    try:
        dashboard.start()
    except KeyboardInterrupt:
        dashboard.stop()


if __name__ == "__main__":
    main()