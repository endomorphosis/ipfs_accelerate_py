#!/usr/bin/env python3
"""
Tests for the Comprehensive Monitoring Dashboard

This module tests the functionality of the monitoring dashboard for
the distributed testing framework.
"""

import os
import sys
import unittest
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Import the monitoring dashboard
    from data.duckdb.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
    
    # Check dependencies
    try:
        import aiohttp
        AIOHTTP_AVAILABLE = True
    except ImportError:
        AIOHTTP_AVAILABLE = False
    
    try:
        from jinja2 import Environment
        JINJA2_AVAILABLE = True
    except ImportError:
        JINJA2_AVAILABLE = False
    
    # Skip test if dependencies are missing
    SKIP_TEST = not (AIOHTTP_AVAILABLE and JINJA2_AVAILABLE)
except ImportError as e:
    print(f"Import error: {e}")
    SKIP_TEST = True


class MockResultAggregator:
    """Mock result aggregator for testing."""
    
    def __init__(self):
        """Initialize with mock data."""
        self.test_analysis = {}
        self.worker_analysis = {}
        self.task_type_analysis = {}
        self.dimension_analysis = {}
        self.regressions = {}
        self.historical_performance = {}
        
        # Initialize with some test data
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Set up test data for the mock result aggregator."""
        # Add test data for test analysis
        self.test_analysis = {
            "test1": {
                "execution_count": 100,
                "success_rate": 0.95,
                "average_duration": 25.3,
                "last_execution": datetime.now() - timedelta(hours=1)
            },
            "test2": {
                "execution_count": 80,
                "success_rate": 0.92,
                "average_duration": 32.1,
                "last_execution": datetime.now() - timedelta(hours=2)
            }
        }
        
        # Add test data for worker analysis
        self.worker_analysis = {
            "worker1": {
                "execution_count": 150,
                "success_rate": 0.97,
                "average_duration": 28.7,
                "task_type_distribution": {
                    "type_counts": {"benchmark": 100, "inference": 50},
                    "type_percentages": {"benchmark": 66.7, "inference": 33.3}
                }
            },
            "worker2": {
                "execution_count": 130,
                "success_rate": 0.94,
                "average_duration": 31.2,
                "task_type_distribution": {
                    "type_counts": {"benchmark": 80, "inference": 50},
                    "type_percentages": {"benchmark": 61.5, "inference": 38.5}
                }
            }
        }
    
    def get_overall_status(self):
        """Get overall system status."""
        return {
            "test_count": len(self.test_analysis),
            "worker_count": len(self.worker_analysis),
            "task_type_count": 2,
            "total_executions": sum(t["execution_count"] for t in self.test_analysis.values()),
            "aggregated_metrics": {
                "throughput_mean": 156.7,
                "latency_mean": 42.3,
                "memory_usage_mean": 2.45,
                "success_rate_mean": 0.95
            },
            "regression_count": 2,
            "significant_regression_count": 1
        }
    
    def get_test_analysis(self):
        """Get test analysis data."""
        return self.test_analysis
    
    def get_dimension_analysis(self, dimension=None):
        """Get dimension analysis data."""
        if dimension:
            return self.dimension_analysis.get(dimension, {})
        return self.dimension_analysis
    
    def get_regressions(self, significant_only=False):
        """Get regression data."""
        if significant_only:
            return {k: v for k, v in self.regressions.items() if v.get("has_significant_regression", False)}
        return self.regressions
    
    def aggregate_results(self, result_type, aggregation_level, filter_params=None):
        """Aggregate results based on parameters."""
        # Return mock aggregated results
        return {
            "results": {
                "basic_statistics": {
                    "model1": {
                        "throughput_items_per_second": {"mean": 150.2, "std": 5.1},
                        "average_latency_ms": {"mean": 28.3, "std": 2.1}
                    },
                    "model2": {
                        "throughput_items_per_second": {"mean": 142.3, "std": 4.8},
                        "average_latency_ms": {"mean": 32.5, "std": 2.4}
                    }
                }
            }
        }
    
    def get_result_anomalies(self, result_type, aggregation_level):
        """Get result anomalies."""
        return {
            "anomalies": [
                {
                    "entity_id": "model1",
                    "metric": "throughput_items_per_second",
                    "value": 150.2,
                    "expected_value": 165.7,
                    "deviation": -9.4,
                    "significance": 0.02
                }
            ]
        }
    
    def get_comparison_report(self, result_type, aggregation_level):
        """Get comparison report."""
        return {
            "comparisons": [
                {
                    "entity_id": "model1",
                    "baseline_period": "2025-03-01/2025-03-03",
                    "current_period": "2025-03-10/2025-03-12",
                    "metrics": {
                        "throughput_items_per_second": {
                            "baseline_mean": 165.7,
                            "current_mean": 150.2,
                            "percent_change": -9.4,
                            "p_value": 0.02
                        }
                    }
                }
            ]
        }


class TestMonitoringDashboard(unittest.TestCase):
    """Tests for the monitoring dashboard."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        if SKIP_TEST:
            return
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = cls.temp_dir.name
        
        # Create mock result aggregator
        cls.result_aggregator = MockResultAggregator()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_initialization(self):
        """Test dashboard initialization."""
        dashboard = MonitoringDashboard(
            host="localhost",
            port=8082,
            coordinator_url=None,
            result_aggregator=self.result_aggregator,
            output_dir=self.output_dir
        )
        
        # Check initialization
        self.assertEqual(dashboard.host, "localhost")
        self.assertEqual(dashboard.port, 8082)
        self.assertEqual(dashboard.output_dir, self.output_dir)
        self.assertEqual(dashboard.result_aggregator, self.result_aggregator)
        
        # Check directory creation
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "visualizations")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "static")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "templates")))
        
        # Check default configuration
        self.assertEqual(dashboard.config["auto_refresh"], 30)
        self.assertEqual(dashboard.config["theme"], "dark")
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_configuration(self):
        """Test dashboard configuration."""
        dashboard = MonitoringDashboard(
            host="localhost",
            port=8082,
            output_dir=self.output_dir
        )
        
        # Test configuration updates
        dashboard.configure({
            "auto_refresh": 60,
            "theme": "light",
            "enable_alerts": False
        })
        
        # Check updated configuration
        self.assertEqual(dashboard.config["auto_refresh"], 60)
        self.assertEqual(dashboard.config["theme"], "light")
        self.assertEqual(dashboard.config["enable_alerts"], False)
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_template_creation(self):
        """Test template creation."""
        dashboard = MonitoringDashboard(
            host="localhost",
            port=8082,
            output_dir=self.output_dir
        )
        
        # Check template creation
        template_dir = os.path.join(self.output_dir, "templates")
        self.assertTrue(os.path.exists(os.path.join(template_dir, "base.html")))
        self.assertTrue(os.path.exists(os.path.join(template_dir, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(template_dir, "system.html")))
        self.assertTrue(os.path.exists(os.path.join(template_dir, "workers.html")))
        self.assertTrue(os.path.exists(os.path.join(template_dir, "tasks.html")))
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_alert_system(self):
        """Test alert system."""
        dashboard = MonitoringDashboard(
            host="localhost",
            port=8082,
            output_dir=self.output_dir
        )
        
        # Add test alerts
        dashboard._add_alert(
            level="critical",
            title="Critical Test Alert",
            message="This is a critical test alert",
            source="test",
            metrics={"value": 100}
        )
        
        dashboard._add_alert(
            level="warning",
            title="Warning Test Alert",
            message="This is a warning test alert",
            source="test",
            metrics={"value": 50}
        )
        
        dashboard._add_alert(
            level="info",
            title="Info Test Alert",
            message="This is an info test alert",
            source="test",
            metrics={"value": 25}
        )
        
        # Check alert history
        self.assertEqual(len(dashboard.alert_history), 3)
        
        # Check alert levels
        levels = [a["level"] for a in dashboard.alert_history]
        self.assertIn("critical", levels)
        self.assertIn("warning", levels)
        self.assertIn("info", levels)
    
    @unittest.skipIf(SKIP_TEST or not AIOHTTP_AVAILABLE, "Missing aiohttp dependency")
    def test_async_start_stop(self):
        """Test async start and stop."""
        dashboard = MonitoringDashboard(
            host="localhost",
            port=8082,
            output_dir=self.output_dir
        )
        
        # Configure for testing
        dashboard.configure({
            "auto_refresh": 0,
            "update_interval": 1,
            "auto_connect_coordinator": False
        })
        
        # Start dashboard in background
        thread = dashboard.start_async()
        
        # Wait for dashboard to start
        time.sleep(2)
        
        # Check that dashboard is running
        self.assertTrue(dashboard.running)
        
        # Stop dashboard
        dashboard.stop()
        
        # Wait for dashboard to stop
        thread.join(timeout=5)
        
        # Check that dashboard is stopped
        self.assertFalse(dashboard.running)
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_get_status_data(self):
        """Test getting status data."""
        dashboard = MonitoringDashboard(
            host="localhost",
            port=8082,
            result_aggregator=self.result_aggregator,
            output_dir=self.output_dir
        )
        
        # Add some test data
        dashboard.worker_connections = {
            "worker1": {"status": "active"},
            "worker2": {"status": "idle"},
            "worker3": {"status": "active"}
        }
        
        dashboard.task_execution_tracking = {
            "task1": {"status": "running"},
            "task2": {"status": "completed"},
            "task3": {"status": "failed"}
        }
        
        dashboard.system_metrics = {
            "health_score": 85,
            "coordinator_status": "active"
        }
        
        # Get status data
        status_data = dashboard._get_status_data()
        
        # Check status data
        self.assertEqual(status_data["workers"]["total"], 3)
        self.assertEqual(status_data["workers"]["active"], 2)
        self.assertEqual(status_data["workers"]["idle"], 1)
        
        self.assertEqual(status_data["tasks"]["total"], 3)
        self.assertEqual(status_data["tasks"]["running"], 1)
        self.assertEqual(status_data["tasks"]["completed"], 1)
        self.assertEqual(status_data["tasks"]["failed"], 1)
        
        self.assertEqual(status_data["system"]["health_score"], 85)
        self.assertEqual(status_data["system"]["coordinator_status"], "active")


if __name__ == "__main__":
    unittest.main()