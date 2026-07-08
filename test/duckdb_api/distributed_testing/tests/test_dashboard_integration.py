#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for the monitoring dashboard.
"""

import os
import sys
import unittest
import tempfile
import json
import time
import threading
import requests
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monitoring_dashboard import MonitoringDashboard, DashboardMetrics


@contextmanager
def start_dashboard_server(host="localhost", port=8081, db_path=None):
    """Start a dashboard server for testing."""
    dashboard = MonitoringDashboard(
        host=host,
        port=port,
        db_path=db_path,
        auto_open=False
    )
    
    # Start the dashboard in a separate thread
    thread = threading.Thread(target=dashboard.start)
    thread.daemon = True
    thread.start()
    
    # Wait for server to start
    time.sleep(1)
    
    try:
        yield dashboard
    finally:
        dashboard.stop()


class TestDashboardIntegration(unittest.TestCase):
    """Integration tests for the monitoring dashboard."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "dashboard_test.db")
        self.host = "localhost"
        self.port = 8082
        self.base_url = f"http://{self.host}:{self.port}"
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_dashboard_server_starts(self):
        """Test that the dashboard server starts and responds to requests."""
        with start_dashboard_server(host=self.host, port=self.port, db_path=self.db_path):
            # Check that the server is running
            response = requests.get(self.base_url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Distributed Testing Dashboard", response.text)
    
    def test_static_files_served(self):
        """Test that static files are served correctly."""
        with start_dashboard_server(host=self.host, port=self.port, db_path=self.db_path):
            # Check CSS file
            response = requests.get(f"{self.base_url}/static/css/dashboard.css")
            self.assertEqual(response.status_code, 200)
            self.assertIn("dashboard-header", response.text)
            
            # Check JavaScript file
            response = requests.get(f"{self.base_url}/static/js/dashboard.js")
            self.assertEqual(response.status_code, 200)
            self.assertIn("connectWebSocket", response.text)
    
    def test_api_endpoints(self):
        """Test API endpoints."""
        with start_dashboard_server(host=self.host, port=self.port, db_path=self.db_path):
            # Check metrics API
            response = requests.get(f"{self.base_url}/api/metrics")
            self.assertEqual(response.status_code, 200)
            metrics = response.json()
            self.assertIsInstance(metrics, dict)
            
            # Check workers API
            response = requests.get(f"{self.base_url}/api/workers")
            self.assertEqual(response.status_code, 200)
            workers = response.json()
            self.assertIsInstance(workers, list)
            
            # Check tasks API
            response = requests.get(f"{self.base_url}/api/tasks")
            self.assertEqual(response.status_code, 200)
            tasks = response.json()
            self.assertIsInstance(tasks, list)
            
            # Check alerts API
            response = requests.get(f"{self.base_url}/api/alerts")
            self.assertEqual(response.status_code, 200)
            alerts = response.json()
            self.assertIsInstance(alerts, list)
            
            # Check visualizations API
            response = requests.get(f"{self.base_url}/api/visualizations")
            self.assertEqual(response.status_code, 200)
            visualizations = response.json()
            self.assertIsInstance(visualizations, dict)
    
    def test_dashboard_metrics_data(self):
        """Test adding data to dashboard metrics and retrieving it."""
        # Create metrics instance
        metrics = DashboardMetrics(db_path=self.db_path)
        
        # Add worker
        worker_id = "worker-1"
        metrics.record_entity(
            entity_id=worker_id,
            entity_type="worker",
            entity_name="Test Worker 1",
            entity_data={"capabilities": ["cpu", "cuda"]},
            status="active"
        )
        
        # Add task
        task_id = "task-1"
        metrics.record_task(
            task_id=task_id,
            task_type="test",
            task_data={"parameters": {"model": "bert"}},
            status="pending",
            worker_id=None
        )
        
        # Add alert
        metrics.record_alert(
            alert_type="test",
            alert_message="Test alert",
            severity="info",
            entity_id=None,
            entity_type=None
        )
        
        # Start dashboard server
        with start_dashboard_server(host=self.host, port=self.port, db_path=self.db_path):
            # Check that data is accessible through the API
            response = requests.get(f"{self.base_url}/api/workers")
            workers = response.json()
            self.assertEqual(len(workers), 1)
            self.assertEqual(workers[0]["entity_id"], worker_id)
            
            response = requests.get(f"{self.base_url}/api/tasks")
            tasks = response.json()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["task_id"], task_id)
            
            response = requests.get(f"{self.base_url}/api/alerts")
            alerts = response.json()
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["alert_type"], "test")
    
    def test_alert_management(self):
        """Test alert management (acknowledge and resolve)."""
        # Create metrics instance
        metrics = DashboardMetrics(db_path=self.db_path)
        
        # Add alert
        alert_id = metrics.record_alert(
            alert_type="test",
            alert_message="Test alert",
            severity="warning",
            entity_id=None,
            entity_type=None
        )
        
        # Start dashboard server
        with start_dashboard_server(host=self.host, port=self.port, db_path=self.db_path):
            # Acknowledge alert
            response = requests.post(
                f"{self.base_url}/api/alerts",
                json={"alert_id": alert_id, "action": "acknowledge"}
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertTrue(result["success"])
            
            # Check that alert is acknowledged
            response = requests.get(f"{self.base_url}/api/alerts?is_acknowledged=1")
            alerts = response.json()
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["id"], alert_id)
            self.assertEqual(alerts[0]["is_acknowledged"], 1)
            
            # Resolve alert
            response = requests.post(
                f"{self.base_url}/api/alerts",
                json={"alert_id": alert_id, "action": "resolve"}
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertTrue(result["success"])
            
            # Check that alert is resolved (not active)
            response = requests.get(f"{self.base_url}/api/alerts?is_active=0")
            alerts = response.json()
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["id"], alert_id)
            self.assertEqual(alerts[0]["is_active"], 0)


if __name__ == "__main__":
    unittest.main()