#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Monitoring Dashboard component.
"""

import unittest
import os
import sys
import json
import time
import tempfile
import threading
import asyncio
import requests
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from monitoring_dashboard import (
    MonitoringDashboard,
    DashboardMetrics
)


class TestDashboardMetrics(unittest.TestCase):
    """Test suite for DashboardMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_metrics.db")
        
        # Initialize metrics
        self.metrics = DashboardMetrics(db_path=self.db_path)

    def tearDown(self):
        """Tear down test fixtures."""
        # Close database connection
        if hasattr(self.metrics, 'conn'):
            self.metrics.conn.close()
        
        # Remove temporary directory
        self.temp_dir.cleanup()

    def test_record_metric(self):
        """Test recording a metric."""
        # Record a metric
        metric_id = self.metrics.record_metric(
            metric_name="test_metric",
            metric_value=42.0,
            metric_type="gauge",
            category="test",
            entity_id="test-entity",
            entity_type="test"
        )
        
        # Verify metric was recorded
        self.assertIsNotNone(metric_id)
        
        # Retrieve the metric
        metrics = self.metrics.get_metric("test_metric")
        
        # Verify retrieval
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["metric_name"], "test_metric")
        self.assertEqual(metrics[0]["metric_value"], 42.0)
        self.assertEqual(metrics[0]["metric_type"], "gauge")
        self.assertEqual(metrics[0]["category"], "test")
        self.assertEqual(metrics[0]["entity_id"], "test-entity")
        self.assertEqual(metrics[0]["entity_type"], "test")

    def test_record_event(self):
        """Test recording an event."""
        # Record an event
        event_data = {"message": "Test event", "detail": "This is a test"}
        event_id = self.metrics.record_event(
            event_type="test_event",
            event_data=event_data,
            entity_id="test-entity",
            entity_type="test",
            severity="info"
        )
        
        # Verify event was recorded
        self.assertIsNotNone(event_id)
        
        # Retrieve the event
        events = self.metrics.get_events(event_type="test_event")
        
        # Verify retrieval
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "test_event")
        self.assertEqual(events[0]["event_data"]["message"], "Test event")
        self.assertEqual(events[0]["entity_id"], "test-entity")
        self.assertEqual(events[0]["entity_type"], "test")
        self.assertEqual(events[0]["severity"], "info")

    def test_record_entity(self):
        """Test recording an entity."""
        # Record an entity
        entity_data = {"hostname": "worker1", "capabilities": ["cpu", "gpu"]}
        entity_id = self.metrics.record_entity(
            entity_id="worker-1",
            entity_type="worker",
            entity_name="Worker 1",
            entity_data=entity_data,
            status="active"
        )
        
        # Verify entity was recorded
        self.assertIsNotNone(entity_id)
        
        # Retrieve the entity
        entities = self.metrics.get_entities(entity_type="worker")
        
        # Verify retrieval
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["entity_id"], "worker-1")
        self.assertEqual(entities[0]["entity_type"], "worker")
        self.assertEqual(entities[0]["entity_name"], "Worker 1")
        self.assertEqual(entities[0]["entity_data"]["hostname"], "worker1")
        self.assertEqual(entities[0]["status"], "active")
        
        # Update the entity
        updated_data = {"hostname": "worker1", "capabilities": ["cpu", "gpu", "tpu"]}
        entity_id = self.metrics.record_entity(
            entity_id="worker-1",
            entity_type="worker",
            entity_name="Worker 1 Updated",
            entity_data=updated_data,
            status="inactive"
        )
        
        # Verify entity was updated
        entities = self.metrics.get_entities(entity_type="worker")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["entity_name"], "Worker 1 Updated")
        self.assertEqual(entities[0]["entity_data"]["capabilities"], ["cpu", "gpu", "tpu"])
        self.assertEqual(entities[0]["status"], "inactive")

    def test_record_task(self):
        """Test recording a task."""
        # Record a task
        task_data = {"model": "bert", "batch_size": 4}
        task_id = self.metrics.record_task(
            task_id="task-1",
            task_type="benchmark",
            task_data=task_data,
            status="pending",
            worker_id="worker-1",
            priority=3
        )
        
        # Verify task was recorded
        self.assertIsNotNone(task_id)
        
        # Retrieve the task
        tasks = self.metrics.get_tasks()
        
        # Verify retrieval
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "task-1")
        self.assertEqual(tasks[0]["task_type"], "benchmark")
        self.assertEqual(tasks[0]["task_data"]["model"], "bert")
        self.assertEqual(tasks[0]["status"], "pending")
        self.assertEqual(tasks[0]["worker_id"], "worker-1")
        self.assertEqual(tasks[0]["priority"], 3)
        
        # Update the task to completed
        task_id = self.metrics.record_task(
            task_id="task-1",
            task_type="benchmark",
            task_data=task_data,
            status="completed",
            worker_id="worker-1",
            priority=3
        )
        
        # Verify task was updated and execution time was recorded
        tasks = self.metrics.get_tasks(status="completed")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["status"], "completed")
        self.assertIsNotNone(tasks[0]["completed_at"])
        self.assertIsNotNone(tasks[0]["execution_time"])

    def test_record_alert(self):
        """Test recording an alert."""
        # Record an alert
        alert_id = self.metrics.record_alert(
            alert_type="test_alert",
            alert_message="Test alert message",
            severity="warning",
            entity_id="test-entity",
            entity_type="test"
        )
        
        # Verify alert was recorded
        self.assertIsNotNone(alert_id)
        
        # Retrieve the alert
        alerts = self.metrics.get_alerts(alert_type="test_alert")
        
        # Verify retrieval
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["alert_type"], "test_alert")
        self.assertEqual(alerts[0]["alert_message"], "Test alert message")
        self.assertEqual(alerts[0]["severity"], "warning")
        self.assertEqual(alerts[0]["entity_id"], "test-entity")
        self.assertEqual(alerts[0]["entity_type"], "test")
        self.assertEqual(alerts[0]["is_active"], 1)
        self.assertEqual(alerts[0]["is_acknowledged"], 0)
        
        # Acknowledge the alert
        self.metrics.acknowledge_alert(alert_id)
        
        # Verify alert was acknowledged
        alerts = self.metrics.get_alerts(alert_id=alert_id)
        self.assertEqual(alerts[0]["is_acknowledged"], 1)
        self.assertIsNotNone(alerts[0]["acknowledged_at"])
        
        # Resolve the alert
        self.metrics.resolve_alert(alert_id)
        
        # Verify alert was resolved
        alerts = self.metrics.get_alerts(alert_id=alert_id)
        self.assertEqual(alerts[0]["is_active"], 0)

    def test_register_metric_calculator(self):
        """Test registering a metric calculator."""
        # Define a test calculator
        def test_calculator():
            return 42
        
        # Register the calculator
        self.metrics.register_metric_calculator("test_calculated_metric", test_calculator)
        
        # Verify registration
        self.assertIn("test_calculated_metric", self.metrics.metric_calculators)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics()
        
        # Verify calculator was called
        self.assertIn("test_calculated_metric", metrics)
        self.assertEqual(metrics["test_calculated_metric"], 42)

    def test_register_visualization_generator(self):
        """Test registering a visualization generator."""
        # Define a test generator
        def test_generator():
            return {"type": "test", "data": "Test visualization"}
        
        # Register the generator
        self.metrics.register_visualization_generator("test_visualization", test_generator)
        
        # Verify registration
        self.assertIn("test_visualization", self.metrics.visualization_generators)
        
        # Generate visualizations
        visualizations = self.metrics.generate_visualizations()
        
        # Verify generator was called
        self.assertIn("test_visualization", visualizations)
        self.assertEqual(visualizations["test_visualization"]["type"], "test")
        self.assertEqual(visualizations["test_visualization"]["data"], "Test visualization")

    def test_count_entities(self):
        """Test counting entities."""
        # Record entities
        self.metrics.record_entity(
            entity_id="worker-1",
            entity_type="worker",
            entity_name="Worker 1",
            entity_data={},
            status="active"
        )
        
        self.metrics.record_entity(
            entity_id="worker-2",
            entity_type="worker",
            entity_name="Worker 2",
            entity_data={},
            status="active"
        )
        
        self.metrics.record_entity(
            entity_id="worker-3",
            entity_type="worker",
            entity_name="Worker 3",
            entity_data={},
            status="inactive"
        )
        
        # Count all workers
        count = self.metrics._count_entities("worker")
        self.assertEqual(count, 3)
        
        # Count active workers
        count = self.metrics._count_entities("worker", status="active")
        self.assertEqual(count, 2)
        
        # Count inactive workers
        count = self.metrics._count_entities("worker", status="inactive")
        self.assertEqual(count, 1)

    def test_count_tasks(self):
        """Test counting tasks."""
        # Record tasks
        self.metrics.record_task(
            task_id="task-1",
            task_type="benchmark",
            task_data={},
            status="pending"
        )
        
        self.metrics.record_task(
            task_id="task-2",
            task_type="benchmark",
            task_data={},
            status="running"
        )
        
        self.metrics.record_task(
            task_id="task-3",
            task_type="benchmark",
            task_data={},
            status="completed"
        )
        
        # Count all tasks
        count = self.metrics._count_tasks()
        self.assertEqual(count, 3)
        
        # Count pending tasks
        count = self.metrics._count_tasks(status="pending")
        self.assertEqual(count, 1)
        
        # Count running tasks
        count = self.metrics._count_tasks(status="running")
        self.assertEqual(count, 1)
        
        # Count completed tasks
        count = self.metrics._count_tasks(status="completed")
        self.assertEqual(count, 1)

    def test_calculate_avg_task_execution_time(self):
        """Test calculating average task execution time."""
        # Create mock tasks with execution times
        cursor = self.metrics.conn.cursor()
        
        # Insert test data
        cursor.execute('''
        INSERT INTO tasks 
        (task_id, task_type, task_data, status, execution_time, created_at, updated_at, completed_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'))
        ''', ("task-1", "benchmark", "{}", "completed", 10.0))
        
        cursor.execute('''
        INSERT INTO tasks 
        (task_id, task_type, task_data, status, execution_time, created_at, updated_at, completed_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'))
        ''', ("task-2", "benchmark", "{}", "completed", 20.0))
        
        cursor.execute('''
        INSERT INTO tasks 
        (task_id, task_type, task_data, status, execution_time, created_at, updated_at, completed_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'))
        ''', ("task-3", "benchmark", "{}", "completed", 30.0))
        
        self.metrics.conn.commit()
        
        # Calculate average execution time
        avg_time = self.metrics._calculate_avg_task_execution_time()
        
        # Verify result
        self.assertEqual(avg_time, 20.0)

    def test_generate_worker_status_chart(self):
        """Test generating worker status chart."""
        # Record entities
        self.metrics.record_entity(
            entity_id="worker-1",
            entity_type="worker",
            entity_name="Worker 1",
            entity_data={},
            status="active"
        )
        
        self.metrics.record_entity(
            entity_id="worker-2",
            entity_type="worker",
            entity_name="Worker 2",
            entity_data={},
            status="active"
        )
        
        self.metrics.record_entity(
            entity_id="worker-3",
            entity_type="worker",
            entity_name="Worker 3",
            entity_data={},
            status="inactive"
        )
        
        # Generate chart
        chart = self.metrics._generate_worker_status_chart()
        
        # Verify result
        self.assertEqual(chart["type"], "plotly")
        self.assertIn("data", chart)


@unittest.skip("Dashboard tests require external dependencies and network access")
class TestMonitoringDashboard(unittest.TestCase):
    """Test suite for MonitoringDashboard class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_dashboard.db")
        
        # Initialize dashboard with test settings
        self.dashboard = MonitoringDashboard(
            host="localhost",
            port=8088,  # Use non-standard port for testing
            db_path=self.db_path,
            auto_open=False
        )
        
        # Start dashboard in a separate thread
        self.dashboard_thread = threading.Thread(target=self.dashboard.start)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
        
        # Wait for dashboard to start
        time.sleep(1)

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop dashboard
        self.dashboard.stop()
        
        # Wait for thread to complete
        self.dashboard_thread.join(timeout=5)
        
        # Remove temporary directory
        self.temp_dir.cleanup()

    def test_dashboard_api_metrics(self):
        """Test dashboard API for metrics."""
        # Record a test metric
        self.dashboard.metrics.record_metric(
            metric_name="test_metric",
            metric_value=42.0,
            metric_type="gauge",
            category="test"
        )
        
        # Request metrics API
        response = requests.get("http://localhost:8088/api/metrics")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("test_metric", data)
        self.assertEqual(data["test_metric"], 42.0)

    def test_dashboard_api_events(self):
        """Test dashboard API for events."""
        # Record a test event
        self.dashboard.metrics.record_event(
            event_type="test_event",
            event_data={"message": "Test event"}
        )
        
        # Request events API
        response = requests.get("http://localhost:8088/api/events")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["event_type"], "test_event")
        self.assertEqual(data[0]["event_data"]["message"], "Test event")

    def test_dashboard_api_tasks(self):
        """Test dashboard API for tasks."""
        # Record a test task
        self.dashboard.metrics.record_task(
            task_id="task-1",
            task_type="benchmark",
            task_data={"model": "bert"},
            status="pending"
        )
        
        # Request tasks API
        response = requests.get("http://localhost:8088/api/tasks")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["task_id"], "task-1")
        self.assertEqual(data[0]["task_type"], "benchmark")
        self.assertEqual(data[0]["task_data"]["model"], "bert")
        self.assertEqual(data[0]["status"], "pending")

    def test_dashboard_api_workers(self):
        """Test dashboard API for workers."""
        # Record a test worker
        self.dashboard.metrics.record_entity(
            entity_id="worker-1",
            entity_type="worker",
            entity_name="Worker 1",
            entity_data={"hostname": "worker1"},
            status="active"
        )
        
        # Request workers API
        response = requests.get("http://localhost:8088/api/workers")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["entity_id"], "worker-1")
        self.assertEqual(data[0]["entity_type"], "worker")
        self.assertEqual(data[0]["entity_name"], "Worker 1")
        self.assertEqual(data[0]["entity_data"]["hostname"], "worker1")
        self.assertEqual(data[0]["status"], "active")

    def test_dashboard_api_alerts(self):
        """Test dashboard API for alerts."""
        # Record a test alert
        self.dashboard.metrics.record_alert(
            alert_type="test_alert",
            alert_message="Test alert message",
            severity="warning"
        )
        
        # Request alerts API
        response = requests.get("http://localhost:8088/api/alerts")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["alert_type"], "test_alert")
        self.assertEqual(data[0]["alert_message"], "Test alert message")
        self.assertEqual(data[0]["severity"], "warning")
        self.assertEqual(data[0]["is_active"], 1)
        self.assertEqual(data[0]["is_acknowledged"], 0)
        
        # Acknowledge alert
        alert_id = data[0]["id"]
        response = requests.post(
            "http://localhost:8088/api/alerts",
            json={"alert_id": alert_id, "action": "acknowledge"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify alert was acknowledged
        response = requests.get("http://localhost:8088/api/alerts")
        data = response.json()
        self.assertEqual(data[0]["is_acknowledged"], 1)
        
        # Resolve alert
        response = requests.post(
            "http://localhost:8088/api/alerts",
            json={"alert_id": alert_id, "action": "resolve"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify alert was resolved
        response = requests.get("http://localhost:8088/api/alerts")
        data = response.json()
        self.assertEqual(data[0]["is_active"], 0)

    def test_dashboard_api_visualizations(self):
        """Test dashboard API for visualizations."""
        # Request visualizations API
        response = requests.get("http://localhost:8088/api/visualizations")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # At least standard visualizations should be present
        self.assertIn("worker_status_chart", data)
        self.assertIn("task_status_chart", data)


if __name__ == '__main__':
    unittest.main()