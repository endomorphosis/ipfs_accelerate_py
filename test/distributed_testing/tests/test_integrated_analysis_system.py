#!/usr/bin/env python3
"""
Unit tests for the Integrated Analysis System

This module contains comprehensive tests for the IntegratedAnalysisSystem,
covering initialization, analysis capabilities, visualization, and integration
with the Distributed Testing Framework Coordinator.
"""

import unittest
import tempfile
import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the IntegratedAnalysisSystem
from result_aggregator.integrated_analysis_system import IntegratedAnalysisSystem

# Global variables for test setup
DUCKDB_AVAILABLE = True
DATA_ANALYSIS_AVAILABLE = True
VISUALIZATION_AVAILABLE = True
STATISTICAL_ANALYSIS_AVAILABLE = True
ML_AVAILABLE = True
PIPELINE_AVAILABLE = True

try:
    import duckdb
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
except ImportError:
    DATA_ANALYSIS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from scipy import stats
except ImportError:
    STATISTICAL_ANALYSIS_AVAILABLE = False


class TestIntegratedAnalysisSystem(unittest.TestCase):
    """Test cases for the IntegratedAnalysisSystem"""

    @classmethod
    def setUpClass(cls):
        """Set up resources needed for all tests"""
        # Skip all tests if critical dependencies aren't available
        if not DUCKDB_AVAILABLE:
            raise unittest.SkipTest("DuckDB not available, skipping tests")

    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary database file
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.duckdb')
        
        # Create test directories
        os.makedirs("test_reports", exist_ok=True)
        os.makedirs("test_visualizations", exist_ok=True)
        
        # Mock coordinator for testing
        self.mock_coordinator = MagicMock()
        self.mock_coordinator.coordinator_id = "test-coordinator"
        
        # Create the analysis system instance
        self.analysis_system = IntegratedAnalysisSystem(
            db_path=self.temp_db_path,
            enable_ml=ML_AVAILABLE,
            enable_visualization=VISUALIZATION_AVAILABLE,
            enable_real_time_analysis=True,
            analysis_interval=timedelta(seconds=1)  # Short interval for testing
        )

    def tearDown(self):
        """Clean up after each test"""
        # Close the analysis system
        self.analysis_system.close()
        
        # Remove temporary database file
        os.close(self.temp_db_fd)
        try:
            os.unlink(self.temp_db_path)
        except:
            pass
            
        # Clean up test directories
        for file in os.listdir("test_reports"):
            os.unlink(os.path.join("test_reports", file))
        for file in os.listdir("test_visualizations"):
            os.unlink(os.path.join("test_visualizations", file))

    def test_initialization(self):
        """Test initialization of the IntegratedAnalysisSystem"""
        self.assertEqual(self.analysis_system.db_path, self.temp_db_path)
        self.assertEqual(self.analysis_system.enable_ml, ML_AVAILABLE)
        self.assertEqual(self.analysis_system.enable_visualization, VISUALIZATION_AVAILABLE)
        self.assertEqual(self.analysis_system.enable_real_time_analysis, True)
        self.assertIsNone(self.analysis_system.coordinator_integration)
        self.assertEqual(len(self.analysis_system.notification_handlers), 0)

    def test_register_with_coordinator(self):
        """Test registration with a coordinator"""
        integration = self.analysis_system.register_with_coordinator(self.mock_coordinator)
        self.assertIsNotNone(integration)
        self.assertIsNotNone(self.analysis_system.coordinator_integration)
        
    def test_notification_handlers(self):
        """Test notification handler registration and removal"""
        # Mock notification handler
        mock_handler = MagicMock()
        
        # Register handler
        result = self.analysis_system.register_notification_handler(mock_handler)
        self.assertTrue(result)
        self.assertEqual(len(self.analysis_system.notification_handlers), 1)
        
        # Test sending a notification
        notification = {
            "type": "test",
            "severity": "info",
            "message": "Test notification"
        }
        self.analysis_system._send_notification(notification)
        mock_handler.assert_called_once_with(notification)
        
        # Unregister handler
        result = self.analysis_system.unregister_notification_handler(mock_handler)
        self.assertTrue(result)
        self.assertEqual(len(self.analysis_system.notification_handlers), 0)
        
        # Try unregistering a non-existent handler
        result = self.analysis_system.unregister_notification_handler(mock_handler)
        self.assertFalse(result)

    def test_store_and_get_result(self):
        """Test storing and retrieving a result"""
        # Create a test result
        test_result = {
            "task_id": "test_task_1",
            "worker_id": "test_worker_1",
            "type": "benchmark",
            "status": "success",
            "metrics": {
                "throughput": 150.5,
                "latency": 6.2,
                "memory_usage": 1024.0
            },
            "details": {
                "model": "bert",
                "batch_size": 8,
                "precision": "fp16"
            }
        }
        
        # Mock the service methods
        self.analysis_system.service.store_result = MagicMock(return_value=1)
        self.analysis_system.service.get_result = MagicMock(return_value=test_result)
        self.analysis_system._analyze_result = MagicMock()
        
        # Store the result
        result_id = self.analysis_system.store_result(test_result)
        self.assertEqual(result_id, 1)
        self.analysis_system.service.store_result.assert_called_once_with(test_result)
        self.analysis_system._analyze_result.assert_called_once_with(1)
        
        # Retrieve the result
        result = self.analysis_system.get_result(1)
        self.assertEqual(result, test_result)
        self.analysis_system.service.get_result.assert_called_once_with(1)

    def test_analyze_results(self):
        """Test analyzing results"""
        # Mock the service methods
        self.analysis_system.get_results = MagicMock(return_value=[
            {"id": 1, "task_id": "task_1", "worker_id": "worker_1", "status": "success", 
             "metrics": {"throughput": 150, "latency": 6}, "type": "benchmark",
             "timestamp": datetime.now().isoformat(), "duration": 10,
             "details": {"hardware": "cuda", "model": "bert", "batch_size": 8}}
        ])
        
        self.analysis_system.service.analyze_performance_trends = MagicMock(return_value={"throughput": {"trend": "stable"}})
        self.analysis_system.service.detect_anomalies = MagicMock(return_value=[])
        
        # If pandas is available, mock the analyze functions
        if DATA_ANALYSIS_AVAILABLE:
            with patch('result_aggregator.integrated_analysis_system.analyze_workload_distribution') as mock_wl:
                with patch('result_aggregator.integrated_analysis_system.analyze_failure_patterns') as mock_fp:
                    with patch('result_aggregator.integrated_analysis_system.analyze_multi_dimensional_performance') as mock_mp:
                        # Set up mock returns
                        mock_wl.return_value = {"worker_stats": {}}
                        mock_fp.return_value = {"failure_counts": {}}
                        mock_mp.return_value = {"dimensions": []}
                        
                        # Call analyze_results
                        results = self.analysis_system.analyze_results(
                            analysis_types=["trends", "anomalies", "workload", "failures", "performance"],
                            metrics=["throughput", "latency"],
                            group_by="hardware",
                            time_period_days=7
                        )
                        
                        # Verify calls
                        self.assertIn("trends", results)
                        self.assertIn("anomalies", results)
                        if DATA_ANALYSIS_AVAILABLE:
                            mock_wl.assert_called_once()
                            mock_fp.assert_called_once()
                            mock_mp.assert_called_once()
        else:
            # Just test basic functionality without pandas
            results = self.analysis_system.analyze_results(
                analysis_types=["trends", "anomalies"],
                metrics=["throughput", "latency"]
            )
            self.assertIn("trends", results)
            self.assertIn("anomalies", results)

    def test_generate_report(self):
        """Test report generation"""
        # Mock the service methods
        mock_report = "# Test Report\n\nThis is a test report."
        self.analysis_system.service.generate_analysis_report = MagicMock(return_value=mock_report)
        self.analysis_system.service.generate_performance_report = MagicMock(return_value=mock_report)
        
        # Generate a report without analysis results
        report = self.analysis_system.generate_report(
            report_type="comprehensive",
            format="markdown",
            output_path="test_reports/test_report.md"
        )
        
        self.assertEqual(report, mock_report)
        self.analysis_system.service.generate_analysis_report.assert_called_once()
        
        # Check if file was created
        self.assertTrue(os.path.exists("test_reports/test_report.md"))
        
        # Generate with pre-computed analysis
        analysis_results = {"trends": {}, "anomalies": []}
        self.analysis_system.analyze_results = MagicMock(return_value=analysis_results)
        
        report = self.analysis_system.generate_report(
            analysis_results=analysis_results,
            report_type="performance",
            format="markdown",
            output_path="test_reports/perf_report.md"
        )
        
        self.assertEqual(report, mock_report)
        self.analysis_system.service.generate_performance_report.assert_called_once()
        self.assertTrue(os.path.exists("test_reports/perf_report.md"))

    @unittest.skipIf(not VISUALIZATION_AVAILABLE, "Visualization packages not available")
    def test_visualize_results(self):
        """Test visualization generation"""
        # Override visualization methods with mocks
        self.analysis_system._visualize_trends = MagicMock(return_value=True)
        self.analysis_system._visualize_anomalies = MagicMock(return_value=True)
        self.analysis_system._visualize_workload_distribution = MagicMock(return_value=True)
        
        # Test trends visualization
        self.analysis_system.visualize_results(
            visualization_type="trends",
            data={"throughput": {"trend": "increasing"}},
            metrics=["throughput"],
            output_path="test_visualizations/trends.png"
        )
        self.analysis_system._visualize_trends.assert_called_once()
        
        # Test anomalies visualization
        self.analysis_system.visualize_results(
            visualization_type="anomalies",
            data=[{"score": 0.9, "type": "outlier"}],
            output_path="test_visualizations/anomalies.png"
        )
        self.analysis_system._visualize_anomalies.assert_called_once()
        
        # Test workload visualization
        self.analysis_system.visualize_results(
            visualization_type="workload_distribution",
            data={"worker_stats": {}},
            output_path="test_visualizations/workload.png"
        )
        self.analysis_system._visualize_workload_distribution.assert_called_once()
        
        # Test with disabled visualization
        temp_viz = self.analysis_system.enable_visualization
        self.analysis_system.enable_visualization = False
        result = self.analysis_system.visualize_results(
            visualization_type="trends",
            data={"throughput": {"trend": "increasing"}},
            metrics=["throughput"]
        )
        self.assertFalse(result)
        self.analysis_system.enable_visualization = temp_viz

    def test_cleanup_old_data(self):
        """Test cleanup of old data"""
        self.analysis_system.service.cleanup_old_data = MagicMock(return_value=5)
        result = self.analysis_system.cleanup_old_data(days=30)
        self.assertEqual(result, 5)
        self.analysis_system.service.cleanup_old_data.assert_called_once_with(30)

    def test_close(self):
        """Test closing the analysis system"""
        # Setup mocks
        self.analysis_system._stop_analysis_thread = MagicMock()
        self.analysis_system.service.close = MagicMock()
        self.analysis_system.coordinator_integration = MagicMock()
        
        # Close the system
        self.analysis_system.close()
        
        # Verify calls
        self.analysis_system._stop_analysis_thread.assert_called_once()
        self.analysis_system.service.close.assert_called_once()
        self.analysis_system.coordinator_integration.close.assert_called_once()


class TestIntegratedAnalysisSystemWithCoordinator(unittest.IsolatedAsyncioTestCase):
    """Test cases for the IntegratedAnalysisSystem with a real coordinator"""

    async def asyncSetUp(self):
        """Set up test environment before each test"""
        if not DUCKDB_AVAILABLE:
            self.skipTest("DuckDB not available, skipping tests")
            
        # Create a temporary database file
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.duckdb')
        
        # Import coordinator
        from coordinator import DistributedTestingCoordinator
        
        # Create the coordinator
        self.coordinator = DistributedTestingCoordinator(
            db_path=self.temp_db_path,
            port=8081,
            enable_advanced_scheduler=True,
            enable_plugins=True
        )
        
        # Create the analysis system instance
        self.analysis_system = IntegratedAnalysisSystem(
            db_path=self.temp_db_path,
            enable_ml=ML_AVAILABLE,
            enable_visualization=VISUALIZATION_AVAILABLE,
            enable_real_time_analysis=True,
            analysis_interval=timedelta(seconds=1)  # Short interval for testing
        )
        
        # Register with coordinator
        self.analysis_system.register_with_coordinator(self.coordinator)
        
        # Create a test worker
        self.worker_id = "test-worker"
        self.coordinator.workers[self.worker_id] = {
            "worker_id": self.worker_id,
            "hostname": "test-host",
            "capabilities": {"hardware": ["cuda", "cpu"]},
            "status": "active",
            "tasks_completed": 0,
            "tasks_failed": 0
        }

    async def asyncTearDown(self):
        """Clean up after each test"""
        # Close the analysis system
        self.analysis_system.close()
        
        # Remove temporary database file
        os.close(self.temp_db_fd)
        try:
            os.unlink(self.temp_db_path)
        except:
            pass

    async def test_task_completion_integration(self):
        """Test integration with task completion"""
        # Create a test task
        task_id = "test-task"
        
        # Create task in coordinator
        self.coordinator.tasks[task_id] = {
            "task_id": task_id,
            "type": "benchmark",
            "status": "running",
            "priority": 1,
            "requirements": {"hardware": ["cuda"]},
            "metadata": {"model": "bert", "batch_size": 8},
            "attempts": 1,
            "started": datetime.now().isoformat()
        }
        
        # Associate task with worker
        self.coordinator.running_tasks[task_id] = self.worker_id
        
        # Create test result
        result = {
            "status": "success",
            "metrics": {
                "throughput": 150.5,
                "latency": 6.2,
                "memory_usage": 1024.0
            },
            "details": {
                "model": "bert",
                "batch_size": 8,
                "precision": "fp16"
            }
        }
        
        # Set up notification receiver
        notifications = []
        def notification_handler(notification):
            notifications.append(notification)
        
        self.analysis_system.register_notification_handler(notification_handler)
        
        # Call the task completion handler
        test_duration = 10.5
        await self.coordinator._handle_task_completed(task_id, self.worker_id, result, test_duration)
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Verify that the task was completed
        self.assertNotIn(task_id, self.coordinator.running_tasks)
        self.assertEqual(self.coordinator.tasks[task_id]["status"], "completed")
        
        # Check for any notifications (we may not get any for a single task)
        # This is mostly to ensure the process completes without errors
        
        # Generate and check a report
        report = self.analysis_system.generate_report(
            report_type="summary",
            format="json"
        )
        
        # Basic validation - should contain valid JSON
        report_data = json.loads(report)
        self.assertIsInstance(report_data, dict)


if __name__ == '__main__':
    unittest.main()