#!/usr/bin/env python3
"""
Unit tests for the Circuit Breaker Visualization components.

These tests verify that the Circuit Breaker visualization correctly
generates dashboard components and metrics.
"""

import os
import sys
import json
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import circuit breaker visualization components
from duckdb_api.distributed_testing.dashboard.circuit_breaker_visualization import (
    CircuitBreakerVisualization, CircuitBreakerDashboardIntegration
)
from duckdb_api.distributed_testing.circuit_breaker import (
    CircuitBreaker, CircuitState
)


class TestCircuitBreakerVisualization(unittest.TestCase):
    """Tests for the CircuitBreakerVisualization class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create visualization
        self.visualization = CircuitBreakerVisualization(output_dir=self.temp_dir)
        
        # Mock metrics data
        self.metrics = {
            "worker_circuits": {
                "worker1": {
                    "name": "worker_worker1",
                    "state": "CLOSED",
                    "health_percentage": 100.0,
                    "failure_count": 0,
                    "total_failures": 0,
                    "total_successes": 10
                },
                "worker2": {
                    "name": "worker_worker2",
                    "state": "OPEN",
                    "health_percentage": 10.0,
                    "failure_count": 3,
                    "total_failures": 5,
                    "total_successes": 15
                }
            },
            "task_circuits": {
                "benchmark": {
                    "name": "task_benchmark",
                    "state": "HALF_OPEN",
                    "health_percentage": 50.0,
                    "failure_count": 0,
                    "total_failures": 3,
                    "total_successes": 8
                }
            },
            "endpoint_circuits": {
                "api/tasks": {
                    "name": "endpoint_api/tasks",
                    "state": "CLOSED",
                    "health_percentage": 95.0,
                    "failure_count": 0,
                    "total_failures": 1,
                    "total_successes": 20
                }
            },
            "global_health": 75.0,
            "last_update": "2025-03-18T10:30:00.000000"
        }
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_generate_circuit_state_indicators(self):
        """Test generation of circuit state indicators."""
        # Generate indicators
        indicators = self.visualization._generate_circuit_state_indicators(self.metrics)
        
        # Check that indicators were generated for each circuit type
        self.assertIn("workers", indicators)
        self.assertIn("tasks", indicators)
        self.assertIn("endpoints", indicators)
        
        # Check worker indicators
        self.assertEqual(len(indicators["workers"]), 2)
        self.assertEqual(indicators["workers"][0]["id"], "worker1")
        self.assertEqual(indicators["workers"][0]["state"], "CLOSED")
        self.assertEqual(indicators["workers"][0]["color"], "green")
        
        self.assertEqual(indicators["workers"][1]["id"], "worker2")
        self.assertEqual(indicators["workers"][1]["state"], "OPEN")
        self.assertEqual(indicators["workers"][1]["color"], "red")
        
        # Check task indicators
        self.assertEqual(len(indicators["tasks"]), 1)
        self.assertEqual(indicators["tasks"][0]["id"], "benchmark")
        self.assertEqual(indicators["tasks"][0]["state"], "HALF_OPEN")
        self.assertEqual(indicators["tasks"][0]["color"], "yellow")
        
        # Check endpoint indicators
        self.assertEqual(len(indicators["endpoints"]), 1)
        self.assertEqual(indicators["endpoints"][0]["id"], "api/tasks")
        self.assertEqual(indicators["endpoints"][0]["state"], "CLOSED")
        self.assertEqual(indicators["endpoints"][0]["color"], "green")
    
    def test_generate_global_health_gauge(self):
        """Test generation of global health gauge."""
        # Generate gauge
        gauge = self.visualization._generate_global_health_gauge(self.metrics)
        
        # Check that gauge was generated
        self.assertIn("figure", gauge)
        self.assertIn("value", gauge)
        self.assertEqual(gauge["value"], 75.0)
    
    def test_generate_state_distribution_chart(self):
        """Test generation of state distribution chart."""
        # Generate chart
        chart = self.visualization._generate_state_distribution_chart(self.metrics)
        
        # Check that chart was generated
        self.assertIn("figure", chart)
        self.assertIn("counts", chart)
        
        # Check state counts
        self.assertEqual(chart["counts"]["CLOSED"], 2)
        self.assertEqual(chart["counts"]["OPEN"], 1)
        self.assertEqual(chart["counts"]["HALF_OPEN"], 1)
    
    def test_generate_failure_rate_chart(self):
        """Test generation of failure rate chart."""
        # Generate chart
        chart = self.visualization._generate_failure_rate_chart(self.metrics)
        
        # Check that chart was generated
        self.assertIn("figure", chart)
        self.assertIn("failure_rates", chart)
        
        # Check failure rates
        self.assertEqual(len(chart["failure_rates"]["workers"]), 2)
        self.assertEqual(len(chart["failure_rates"]["tasks"]), 1)
        self.assertEqual(len(chart["failure_rates"]["endpoints"]), 1)
    
    def test_update_history(self):
        """Test updating historical data."""
        # Update history with metrics
        self.visualization._update_history(self.metrics)
        
        # Check that history was updated
        self.assertEqual(len(self.visualization.state_history), 1)
        self.assertEqual(len(self.visualization.metric_history), 1)
        
        # Check state history
        self.assertEqual(self.visualization.state_history[0]["states"]["CLOSED"], 2)
        self.assertEqual(self.visualization.state_history[0]["states"]["OPEN"], 1)
        self.assertEqual(self.visualization.state_history[0]["states"]["HALF_OPEN"], 1)
        
        # Check metric history
        self.assertEqual(self.visualization.metric_history[0]["global_health"], 75.0)
    
    def test_generate_history_chart(self):
        """Test generation of history chart."""
        # Update history with metrics
        self.visualization._update_history(self.metrics)
        
        # Generate chart
        chart = self.visualization._generate_history_chart()
        
        # Check that chart was generated
        self.assertIn("figure", chart)
    
    def test_generate_dashboard(self):
        """Test generation of complete dashboard."""
        # Mock Jinja2 template
        mock_template = MagicMock()
        mock_template.render = MagicMock(return_value="Dashboard HTML")
        self.visualization.env.get_template = MagicMock(side_effect=lambda name: mock_template)
        
        # Generate dashboard
        html = self.visualization.generate_dashboard(self.metrics)
        
        # Check that dashboard was generated
        self.assertEqual(html, "Dashboard HTML")
        
        # Check that template was used
        mock_template.render.assert_called_once()
        
        # Check that dashboard file was created
        dashboard_file = os.path.join(self.temp_dir, "circuit_breaker_dashboard.html")
        self.assertTrue(os.path.exists(dashboard_file))
    
    def test_generate_iframe_html(self):
        """Test generation of iframe HTML."""
        # Generate iframe HTML
        iframe_html = self.visualization.generate_iframe_html()
        
        # Check iframe HTML
        self.assertIn('<iframe src="dashboards/circuit_breakers/circuit_breaker_dashboard.html"', iframe_html)
        self.assertIn('width="100%"', iframe_html)
        self.assertIn('height="800px"', iframe_html)
    
    def test_get_latest_metrics_summary(self):
        """Test getting latest metrics summary."""
        # Update history with metrics
        self.visualization._update_history(self.metrics)
        
        # Get summary
        summary = self.visualization.get_latest_metrics_summary()
        
        # Check summary
        self.assertIn("global_health", summary)
        self.assertIn("worker_health", summary)
        self.assertIn("task_health", summary)
        self.assertIn("endpoint_health", summary)
        self.assertIn("state_counts", summary)


class TestCircuitBreakerDashboardIntegration(unittest.TestCase):
    """Tests for the CircuitBreakerDashboardIntegration class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock coordinator
        self.coordinator = MagicMock()
        
        # Create mock circuit breaker integration
        self.coordinator.circuit_breaker_integration = MagicMock()
        self.coordinator.circuit_breaker_integration.get_circuit_breaker_metrics = MagicMock(return_value={
            "worker_circuits": {},
            "task_circuits": {},
            "endpoint_circuits": {},
            "global_health": 100.0,
            "last_update": "2025-03-18T10:30:00.000000"
        })
        
        # Create dashboard integration
        self.dashboard_integration = CircuitBreakerDashboardIntegration(
            coordinator=self.coordinator,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of the dashboard integration."""
        # Check that dashboard integration was initialized correctly
        self.assertEqual(self.dashboard_integration.coordinator, self.coordinator)
        self.assertEqual(self.dashboard_integration.output_dir, self.temp_dir)
        self.assertIsNotNone(self.dashboard_integration.visualization)
        self.assertEqual(self.dashboard_integration.circuit_breaker_integration, self.coordinator.circuit_breaker_integration)
    
    def test_get_circuit_breaker_metrics(self):
        """Test getting circuit breaker metrics."""
        # Get metrics
        metrics = self.dashboard_integration.get_circuit_breaker_metrics()
        
        # Check that metrics were retrieved
        self.assertIn("worker_circuits", metrics)
        self.assertIn("task_circuits", metrics)
        self.assertIn("endpoint_circuits", metrics)
        self.assertIn("global_health", metrics)
        self.assertIn("last_update", metrics)
        
        # Check that circuit breaker integration was used
        self.coordinator.circuit_breaker_integration.get_circuit_breaker_metrics.assert_called_once()
    
    def test_generate_dashboard(self):
        """Test generating dashboard."""
        # Mock visualization
        self.dashboard_integration.visualization.generate_dashboard = MagicMock(return_value="Dashboard HTML")
        
        # Generate dashboard
        html = self.dashboard_integration.generate_dashboard()
        
        # Check that dashboard was generated
        self.assertEqual(html, "Dashboard HTML")
        
        # Check that visualization was used
        self.dashboard_integration.visualization.generate_dashboard.assert_called_once()
    
    def test_get_dashboard_iframe_html(self):
        """Test getting dashboard iframe HTML."""
        # Mock visualization
        self.dashboard_integration.visualization.generate_iframe_html = MagicMock(return_value="<iframe></iframe>")
        
        # Get iframe HTML
        iframe_html = self.dashboard_integration.get_dashboard_iframe_html()
        
        # Check iframe HTML
        self.assertEqual(iframe_html, "<iframe></iframe>")
        
        # Check that visualization was used
        self.dashboard_integration.visualization.generate_iframe_html.assert_called_once()
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        # Mock visualization
        self.dashboard_integration.visualization.get_latest_metrics_summary = MagicMock(return_value={"global_health": 100.0})
        
        # Get summary
        summary = self.dashboard_integration.get_metrics_summary()
        
        # Check summary
        self.assertEqual(summary, {"global_health": 100.0})
        
        # Check that visualization was used
        self.dashboard_integration.visualization.get_latest_metrics_summary.assert_called_once()


if __name__ == "__main__":
    unittest.main()