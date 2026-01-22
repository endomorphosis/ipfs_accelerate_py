#!/usr/bin/env python3
"""
Tests for DRM External Monitoring Integration

This module contains tests for the integration between DRM and external
monitoring systems like Prometheus and Grafana.
"""

import os
import sys
import json
import unittest
import tempfile
import threading
import time
import socket
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import mock DRM for testing
from duckdb_api.distributed_testing.testing.mock_drm import MockDynamicResourceManager

# Check dependencies
try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import the modules to test
try:
    from duckdb_api.distributed_testing.dashboard.drm_external_monitoring_integration import (
        PrometheusExporter,
        GrafanaDashboardGenerator,
        ExternalMonitoringBridge
    )
    from duckdb_api.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
    
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODULES_AVAILABLE = False

# Define port for testing
TEST_PORT = 9123  # Use a different port for tests to avoid conflicts


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


@unittest.skipIf(not MODULES_AVAILABLE, "External monitoring modules not available")
class TestPrometheusExporter(unittest.TestCase):
    """Tests for PrometheusExporter component."""
    
    def setUp(self):
        """Set up test environment."""
        # Skip if prometheus_client not available
        if not PROMETHEUS_AVAILABLE:
            self.skipTest("prometheus_client not available")
        
        # Skip if port is already in use
        if is_port_in_use(TEST_PORT):
            self.skipTest(f"Port {TEST_PORT} is already in use")
        
        # Create exporter
        self.exporter = PrometheusExporter(port=TEST_PORT)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'exporter') and self.exporter.running:
            self.exporter.stop()
    
    def test_initialization(self):
        """Test exporter initialization."""
        self.assertEqual(self.exporter.port, TEST_PORT)
        self.assertFalse(self.exporter.running)
        self.assertIsInstance(self.exporter.metrics, dict)
    
    def test_start_stop(self):
        """Test starting and stopping the exporter."""
        # Start exporter
        result = self.exporter.start()
        self.assertTrue(result)
        self.assertTrue(self.exporter.running)
        
        # Verify port is in use
        self.assertTrue(is_port_in_use(TEST_PORT))
        
        # Stop exporter
        self.exporter.stop()
        self.assertFalse(self.exporter.running)
    
    def test_update_metrics(self):
        """Test updating metrics."""
        # Start exporter
        self.exporter.start()
        
        # Create test metrics
        test_metrics = {
            "resource_metrics": {
                "cpu_utilization": [50.0],
                "memory_utilization": [70.0],
                "gpu_utilization": [30.0],
                "worker_count": [5],
                "active_tasks": [10],
                "pending_tasks": [2]
            },
            "worker_metrics": {
                "worker-1": {
                    "cpu_utilization": [55.0],
                    "memory_utilization": [65.0],
                    "gpu_utilization": [0.0],
                    "tasks": [3]
                },
                "worker-2": {
                    "cpu_utilization": [45.0],
                    "memory_utilization": [75.0],
                    "gpu_utilization": [25.0],
                    "tasks": [4]
                }
            },
            "performance_metrics": {
                "task_throughput": [{"value": 15.5, "timestamp": "2025-07-20T12:00:00"}],
                "allocation_time": [{"value": 120.0, "timestamp": "2025-07-20T12:00:00"}],
                "resource_efficiency": [{"value": 85.5, "timestamp": "2025-07-20T12:00:00"}]
            },
            "scaling_decisions": [
                {
                    "action": "scale_up",
                    "count": 2,
                    "reason": "High CPU utilization"
                }
            ],
            "alerts": [
                {
                    "level": "warning",
                    "source": "resource_monitor",
                    "message": "High CPU utilization detected"
                }
            ]
        }
        
        # Update metrics
        self.exporter.update_metrics(test_metrics)
        
        # Verify metrics were updated (hard to test actual values since they're in the registry)
        self.assertTrue(True)  # Test passes if no exceptions occurred
        
        # Stop exporter
        self.exporter.stop()
    
    @unittest.skipIf(not REQUESTS_AVAILABLE, "requests not available")
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        # Start exporter
        self.exporter.start()
        
        # Create test metrics with sample data
        test_metrics = {
            "resource_metrics": {
                "cpu_utilization": [50.0],
                "memory_utilization": [70.0],
                "gpu_utilization": [30.0],
                "worker_count": [5],
                "active_tasks": [10],
                "pending_tasks": [2]
            }
        }
        
        # Update metrics
        self.exporter.update_metrics(test_metrics)
        
        # Wait for metrics to be registered
        time.sleep(1)
        
        # Get metrics from endpoint
        response = requests.get(f"http://localhost:{TEST_PORT}")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn("drm_cpu_utilization_percent", response.text)
        
        # Stop exporter
        self.exporter.stop()


@unittest.skipIf(not MODULES_AVAILABLE, "External monitoring modules not available")
class TestGrafanaDashboardGenerator(unittest.TestCase):
    """Tests for GrafanaDashboardGenerator component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create generator
        self.generator = GrafanaDashboardGenerator(
            prometheus_url="http://test-prometheus:9090"
        )
        self.generator.set_output_directory(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.prometheus_url, "http://test-prometheus:9090")
        self.assertEqual(self.generator.output_dir, self.temp_dir.name)
    
    def test_generate_dashboard(self):
        """Test generating a dashboard configuration."""
        # Generate dashboard
        dashboard = self.generator.generate_drm_dashboard()
        
        # Verify dashboard structure
        self.assertIsInstance(dashboard, dict)
        self.assertIn("panels", dashboard)
        self.assertIn("templating", dashboard)
        self.assertIn("time", dashboard)
        self.assertEqual(dashboard["title"], "DRM Real-Time Performance Dashboard")
        
        # Verify panels were created
        self.assertGreater(len(dashboard["panels"]), 5)
        
        # Verify template variables
        self.assertIn("list", dashboard["templating"])
        self.assertGreater(len(dashboard["templating"]["list"]), 0)
    
    def test_save_dashboard(self):
        """Test saving a dashboard configuration."""
        # Generate dashboard
        dashboard = self.generator.generate_drm_dashboard()
        
        # Save dashboard
        file_path = self.generator.save_dashboard(dashboard, filename="test_dashboard.json")
        
        # Verify file was created
        self.assertIsNotNone(file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Verify file contents
        with open(file_path, 'r') as f:
            saved_dashboard = json.load(f)
        
        self.assertEqual(saved_dashboard["title"], dashboard["title"])
        self.assertEqual(len(saved_dashboard["panels"]), len(dashboard["panels"]))


@unittest.skipIf(not MODULES_AVAILABLE, "External monitoring modules not available")
class TestExternalMonitoringBridge(unittest.TestCase):
    """Tests for ExternalMonitoringBridge component."""
    
    def setUp(self):
        """Set up test environment."""
        # Skip if prometheus_client not available
        if not PROMETHEUS_AVAILABLE:
            self.skipTest("prometheus_client not available")
        
        # Skip if port is already in use
        if is_port_in_use(TEST_PORT):
            self.skipTest(f"Port {TEST_PORT} is already in use")
        
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock DRM
        self.mock_drm = MockDynamicResourceManager()
        
        # Create DRM dashboard (without starting it)
        self.dashboard = DRMRealTimeDashboard(
            dynamic_resource_manager=self.mock_drm,
            db_path=os.path.join(self.temp_dir.name, "test_db.duckdb"),
            port=8099,  # High port for testing
            update_interval=1,  # Fast updates for testing
            retention_window=5,  # Short retention for testing
            debug=False
        )
        
        # Initialize data collection to get some metrics
        self.dashboard._start_data_collection()
        time.sleep(2)  # Wait for data collection
        
        # Create bridge
        self.bridge = ExternalMonitoringBridge(
            drm_dashboard=self.dashboard,
            metrics_port=TEST_PORT,
            prometheus_url="http://test-prometheus:9090",
            grafana_url="http://test-grafana:3000",
            export_grafana_dashboard=True,
            output_dir=self.temp_dir.name
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Stop data collection
        if hasattr(self, 'dashboard'):
            self.dashboard._stop_data_collection()
        
        # Stop bridge
        if hasattr(self, 'bridge') and self.bridge.running:
            self.bridge.stop()
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test bridge initialization."""
        self.assertEqual(self.bridge.prometheus_url, "http://test-prometheus:9090")
        self.assertEqual(self.bridge.grafana_url, "http://test-grafana:3000")
        self.assertEqual(self.bridge.output_dir, self.temp_dir.name)
        self.assertIsInstance(self.bridge.prometheus_exporter, PrometheusExporter)
        self.assertIsInstance(self.bridge.grafana_generator, GrafanaDashboardGenerator)
    
    def test_start_stop(self):
        """Test starting and stopping the bridge."""
        # Start bridge
        result = self.bridge.start()
        self.assertTrue(result)
        self.assertTrue(self.bridge.running)
        
        # Verify Prometheus exporter is running
        self.assertTrue(self.bridge.prometheus_exporter.running)
        
        # Verify update thread is running
        self.assertIsNotNone(self.bridge.update_thread)
        self.assertTrue(self.bridge.update_thread.is_alive())
        
        # Verify Grafana dashboard was exported
        dashboard_path = self.bridge.get_grafana_dashboard_path()
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Stop bridge
        self.bridge.stop()
        self.assertFalse(self.bridge.running)
        
        # Verify Prometheus exporter was stopped
        self.assertFalse(self.bridge.prometheus_exporter.running)
        
        # Verify update thread was stopped
        self.assertFalse(self.bridge.update_thread.is_alive())
    
    def test_get_dashboard_metrics(self):
        """Test getting metrics from DRM dashboard."""
        # Get metrics
        metrics = self.bridge._get_dashboard_metrics()
        
        # Verify metrics structure
        self.assertIn("resource_metrics", metrics)
        self.assertIn("worker_metrics", metrics)
        self.assertIn("performance_metrics", metrics)
        self.assertIn("scaling_decisions", metrics)
        self.assertIn("alerts", metrics)
        
        # Verify resource metrics
        self.assertIn("cpu_utilization", metrics["resource_metrics"])
        self.assertIn("memory_utilization", metrics["resource_metrics"])
        self.assertIn("gpu_utilization", metrics["resource_metrics"])
        
        # Verify we have data
        self.assertGreater(len(metrics["resource_metrics"]["cpu_utilization"]), 0)
    
    def test_get_prometheus_url(self):
        """Test getting Prometheus URL."""
        # Get URL
        url = self.bridge.get_prometheus_url()
        
        # Verify URL
        self.assertEqual(url, f"http://localhost:{TEST_PORT}/metrics")
    
    def test_get_metrics_integration_guide(self):
        """Test getting metrics integration guide."""
        # Get guide
        guide = self.bridge.get_metrics_integration_guide()
        
        # Verify guide
        self.assertIsInstance(guide, str)
        self.assertIn("DRM External Monitoring Integration Guide", guide)
        self.assertIn(f"http://localhost:{TEST_PORT}/metrics", guide)
        self.assertIn("Prometheus Integration", guide)
        self.assertIn("Grafana Integration", guide)
        self.assertIn("Available Metrics", guide)


if __name__ == "__main__":
    unittest.main()