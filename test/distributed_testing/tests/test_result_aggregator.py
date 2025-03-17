#!/usr/bin/env python3
"""
Unit tests for the Result Aggregator Service

This module contains tests for the Result Aggregator Service components and coordinator integration.

Usage:
    python -m unittest tests/test_result_aggregator.py
"""

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Import the components for testing
from result_aggregator.service import ResultAggregatorService
from result_aggregator.coordinator_integration import ResultAggregatorIntegration

class TestResultAggregatorService(unittest.TestCase):
    """Test cases for the Result Aggregator Service."""
    
    def setUp(self):
        """Set up the test environment."""
        # Use a temporary database file for testing
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".duckdb")
        
        # Initialize service with test database
        self.service = ResultAggregatorService(
            db_path=self.db_path,
            enable_ml=True,
            enable_visualization=False  # Disable visualization for testing
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Close service
        self.service.close()
        
        # Close and remove temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_store_result(self):
        """Test storing a test result."""
        # Create a test result
        result = {
            "task_id": "test_task_1",
            "worker_id": "test_worker_1",
            "timestamp": datetime.now().isoformat(),
            "type": "benchmark",
            "status": "completed",
            "duration": 10.5,
            "metrics": {
                "throughput": 120.5,
                "latency": 5.2,
                "memory_usage": 1024.0
            },
            "details": {
                "model": "test_model",
                "batch_size": 8,
                "precision": "fp16"
            }
        }
        
        # Store the result
        result_id = self.service.store_result(result)
        
        # Verify result was stored
        self.assertGreater(result_id, 0)
        
        # Retrieve the result and verify
        stored_result = self.service.get_result(result_id)
        self.assertEqual(stored_result["task_id"], result["task_id"])
        self.assertEqual(stored_result["worker_id"], result["worker_id"])
        self.assertEqual(stored_result["type"], result["type"])
        self.assertEqual(stored_result["status"], result["status"])
        self.assertEqual(stored_result["duration"], result["duration"])
        
        # Verify metrics were stored
        self.assertIn("metrics", stored_result)
        self.assertEqual(stored_result["metrics"]["throughput"], result["metrics"]["throughput"])
        self.assertEqual(stored_result["metrics"]["latency"], result["metrics"]["latency"])
        self.assertEqual(stored_result["metrics"]["memory_usage"], result["metrics"]["memory_usage"])
        
        # Verify details were stored
        self.assertIn("details", stored_result)
        self.assertEqual(stored_result["details"]["model"], result["details"]["model"])
        self.assertEqual(stored_result["details"]["batch_size"], result["details"]["batch_size"])
        self.assertEqual(stored_result["details"]["precision"], result["details"]["precision"])
    
    def test_get_results(self):
        """Test retrieving multiple test results with filtering."""
        # Store multiple test results
        for i in range(10):
            result = {
                "task_id": f"test_task_{i}",
                "worker_id": f"test_worker_{i % 3}",
                "timestamp": datetime.now().isoformat(),
                "type": "benchmark" if i % 2 == 0 else "unit_test",
                "status": "completed",
                "duration": 10.5 + i,
                "metrics": {
                    "throughput": 120.5 + i * 10,
                    "latency": 5.2 - i * 0.2,
                    "memory_usage": 1024.0 + i * 100
                },
                "details": {
                    "model": f"test_model_{i % 3}",
                    "batch_size": 8 * (i % 4 + 1),
                    "precision": "fp16" if i % 2 == 0 else "fp32"
                }
            }
            self.service.store_result(result)
        
        # Test filtering by test_type
        benchmark_results = self.service.get_results(
            filter_criteria={"test_type": "benchmark"}
        )
        self.assertEqual(len(benchmark_results), 5)
        for result in benchmark_results:
            self.assertEqual(result["type"], "benchmark")
        
        # Test filtering by worker_id
        worker_results = self.service.get_results(
            filter_criteria={"worker_id": "test_worker_0"}
        )
        for result in worker_results:
            self.assertEqual(result["worker_id"], "test_worker_0")
        
        # Test filtering by status
        completed_results = self.service.get_results(
            filter_criteria={"status": "completed"}
        )
        self.assertEqual(len(completed_results), 10)
        
        # Test pagination
        paginated_results = self.service.get_results(limit=3, offset=2)
        self.assertEqual(len(paginated_results), 3)
        
        # Verify the correct results were returned based on sorting by timestamp
        # (Results are sorted by timestamp DESC, so we should get the 3rd, 4th, and 5th most recent)
        self.assertEqual(paginated_results[0]["task_id"], "test_task_7")
        self.assertEqual(paginated_results[1]["task_id"], "test_task_6")
        self.assertEqual(paginated_results[2]["task_id"], "test_task_5")
    
    def test_aggregated_results(self):
        """Test getting aggregated results."""
        # Store multiple test results
        for i in range(10):
            result = {
                "task_id": f"test_task_{i}",
                "worker_id": f"test_worker_{i % 3}",
                "timestamp": datetime.now().isoformat(),
                "type": "benchmark",
                "status": "completed",
                "duration": 10.5 + i,
                "metrics": {
                    "throughput": 120.5 + i * 10,
                    "latency": 5.2 - i * 0.2,
                    "memory_usage": 1024.0 + i * 100
                },
                "details": {
                    "model": f"test_model_{i % 3}",
                    "batch_size": 8 * (i % 4 + 1),
                    "precision": "fp16" if i % 2 == 0 else "fp32"
                }
            }
            self.service.store_result(result)
        
        # Test aggregation with mean
        aggregated = self.service.get_aggregated_results(
            aggregation_type="mean"
        )
        
        # Verify metrics were aggregated correctly
        self.assertIn("throughput", aggregated)
        self.assertIn("latency", aggregated)
        self.assertIn("memory_usage", aggregated)
        
        # Calculate expected values
        expected_throughput = sum([120.5 + i * 10 for i in range(10)]) / 10
        expected_latency = sum([5.2 - i * 0.2 for i in range(10)]) / 10
        expected_memory = sum([1024.0 + i * 100 for i in range(10)]) / 10
        
        # Verify aggregated values (with some tolerance for floating point)
        self.assertAlmostEqual(aggregated["throughput"], expected_throughput, places=2)
        self.assertAlmostEqual(aggregated["latency"], expected_latency, places=2)
        self.assertAlmostEqual(aggregated["memory_usage"], expected_memory, places=2)
        
        # Test grouping by worker_id
        grouped = self.service.get_aggregated_results(
            aggregation_type="mean",
            group_by=["worker_id"]
        )
        
        # Verify each worker's metrics were aggregated correctly
        self.assertEqual(len(grouped), 3)  # 3 workers
        
        # Verify metrics for each worker
        for group in grouped:
            worker_id = group["worker_id"]
            worker_num = int(worker_id.split("_")[1])
            
            # Just basic verification that the grouping worked
            self.assertIn("throughput", group)
            self.assertIn("latency", group)
            self.assertIn("memory_usage", group)
    
    def test_performance_trends(self):
        """Test performance trend analysis."""
        # Store test results with trending metrics
        for i in range(10):
            # Use a timestamp with increasing time
            timestamp = (datetime.now() - timedelta(hours=10-i)).isoformat()
            
            # Create trending metrics
            throughput = 100.0 + i * 5  # Increasing
            latency = 10.0 - i * 0.5    # Decreasing
            memory = 1000.0             # Stable
            
            result = {
                "task_id": f"trend_task_{i}",
                "worker_id": "test_worker_1",
                "timestamp": timestamp,
                "type": "benchmark",
                "status": "completed",
                "duration": 10.0,
                "metrics": {
                    "throughput": throughput,
                    "latency": latency,
                    "memory_usage": memory
                },
                "details": {
                    "model": "trend_model",
                    "batch_size": 8,
                    "precision": "fp16"
                }
            }
            self.service.store_result(result)
        
        # Analyze trends
        trends = self.service.analyze_performance_trends()
        
        # Verify all metrics were analyzed
        self.assertIn("throughput", trends)
        self.assertIn("latency", trends)
        self.assertIn("memory_usage", trends)
        
        # Verify trend detection worked correctly
        self.assertEqual(trends["throughput"]["trend"], "increasing")
        self.assertEqual(trends["latency"]["trend"], "decreasing")
        
        # Verify memory usage was detected as stable
        # (This might not be exactly "stable" due to random variation, but should not be strongly trending)
        self.assertNotEqual(trends["memory_usage"]["trend"], "increasing")
        
        # Verify percent change is roughly 50% for throughput (100 to 145)
        self.assertAlmostEqual(trends["throughput"]["percent_change"], 45.0, delta=5)
        
        # Verify percent change is roughly -45% for latency (10 to 5.5)
        self.assertAlmostEqual(trends["latency"]["percent_change"], -45.0, delta=5)
    
    def test_report_generation(self):
        """Test report generation."""
        # Store some test results
        for i in range(5):
            result = {
                "task_id": f"report_task_{i}",
                "worker_id": f"test_worker_{i % 2}",
                "timestamp": datetime.now().isoformat(),
                "type": "benchmark",
                "status": "completed",
                "duration": 10.5 + i,
                "metrics": {
                    "throughput": 120.5 + i * 10,
                    "latency": 5.2 - i * 0.2,
                    "memory_usage": 1024.0 + i * 100
                },
                "details": {
                    "model": "report_model",
                    "batch_size": 8,
                    "precision": "fp16"
                }
            }
            self.service.store_result(result)
        
        # Generate a performance report in Markdown format
        report = self.service.generate_analysis_report(
            report_type="performance",
            format="markdown"
        )
        
        # Verify the report is a non-empty string
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Verify it contains markdown headers
        self.assertIn("# Analysis Report", report)
        self.assertIn("## ", report)
        
        # Verify it contains metric names
        self.assertIn("throughput", report)
        self.assertIn("latency", report)
        self.assertIn("memory_usage", report)
        
        # Generate a summary report
        summary_report = self.service.generate_analysis_report(
            report_type="summary",
            format="markdown"
        )
        
        # Verify the summary report is a non-empty string
        self.assertIsInstance(summary_report, str)
        self.assertGreater(len(summary_report), 0)
        
        # Verify it contains summary information
        self.assertIn("Summary Statistics", summary_report)
        self.assertIn("Results by Status", summary_report)
        
        # Generate a performance report in HTML format
        html_report = self.service.generate_analysis_report(
            report_type="performance",
            format="html"
        )
        
        # Verify the HTML report is a non-empty string
        self.assertIsInstance(html_report, str)
        self.assertGreater(len(html_report), 0)
        
        # Verify it contains HTML markup
        self.assertIn("<!DOCTYPE html>", html_report)
        self.assertIn("<html>", html_report)
        self.assertIn("</html>", html_report)
        
        # Verify it contains metric names
        self.assertIn("throughput", html_report)
        self.assertIn("latency", html_report)
        self.assertIn("memory_usage", html_report)
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # First, store normal results to establish a baseline
        for i in range(15):
            result = {
                "task_id": f"normal_task_{i}",
                "worker_id": "test_worker_1",
                "timestamp": datetime.now().isoformat(),
                "type": "benchmark",
                "status": "completed",
                "duration": 10.0 + np.random.normal(0, 1),  # Add small random variation
                "metrics": {
                    "throughput": 100.0 + np.random.normal(0, 5),
                    "latency": 10.0 + np.random.normal(0, 0.5),
                    "memory_usage": 1000.0 + np.random.normal(0, 50)
                },
                "details": {
                    "model": "anomaly_model",
                    "batch_size": 8,
                    "precision": "fp16"
                }
            }
            self.service.store_result(result)
        
        # Then, store an anomalous result
        anomaly_result = {
            "task_id": "anomaly_task",
            "worker_id": "test_worker_1",
            "timestamp": datetime.now().isoformat(),
            "type": "benchmark",
            "status": "completed",
            "duration": 30.0,  # 3x normal
            "metrics": {
                "throughput": 10.0,  # 10x lower than normal
                "latency": 50.0,     # 5x higher than normal
                "memory_usage": 5000.0  # 5x higher than normal
            },
            "details": {
                "model": "anomaly_model",
                "batch_size": 8,
                "precision": "fp16"
            }
        }
        anomaly_id = self.service.store_result(anomaly_result)
        
        # Detect anomalies
        anomalies = self.service.service._detect_anomalies_for_result(anomaly_id)
        
        # Verify anomalies were detected
        self.assertGreater(len(anomalies), 0)
        
        # Verify anomaly score is high
        self.assertGreater(anomalies[0]["score"], 0.7)
        
        # Verify anomalous features were detected
        self.assertIn("anomalous_features", anomalies[0]["details"])
        self.assertGreater(len(anomalies[0]["details"]["anomalous_features"]), 0)
        
        # Check if at least some of our anomalous metrics were detected
        detected_features = [f["feature"] for f in anomalies[0]["details"]["anomalous_features"]]
        self.assertTrue(any(feature in detected_features for feature in ["throughput", "latency", "memory_usage", "duration"]))
        
        # Test batch anomaly detection
        batch_anomalies = self.service.detect_anomalies()
        
        # Verify anomalies were detected
        self.assertGreater(len(batch_anomalies), 0)


class TestResultAggregatorIntegration(unittest.TestCase):
    """Test cases for the Result Aggregator integration with the Coordinator."""
    
    def setUp(self):
        """Set up the test environment."""
        # Use a temporary database file for testing
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".duckdb")
        
        # Create a mock coordinator
        self.coordinator = MagicMock()
        
        # Initialize fake data
        self.coordinator.tasks = {}
        self.coordinator.workers = {}
        self.coordinator.running_tasks = {}
        self.coordinator.completed_tasks = set()
        
        # Mock handle_task_completed method
        self.coordinator._handle_task_completed = AsyncMock()
        
        # Mock handle_task_failed method
        self.coordinator._handle_task_failed = AsyncMock()
        
        # Initialize integration with test database
        self.integration = ResultAggregatorIntegration(
            coordinator=self.coordinator,
            db_path=self.db_path,
            enable_ml=True,
            enable_visualization=False  # Disable visualization for testing
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Close integration
        self.integration.close()
        
        # Close and remove temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_coordinator_integration(self):
        """Test integration with the coordinator."""
        # Register with coordinator
        self.integration.register_with_coordinator()
        
        # Verify registration
        self.assertTrue(self.integration.registered)
        
        # Verify handle_task_completed was patched
        self.assertNotEqual(
            self.coordinator._handle_task_completed._mock_wraps,
            self.integration._original_handle_task_completed
        )
    
    @patch('result_aggregator.coordinator_integration.ResultAggregatorService.store_result')
    async def test_task_completed_hook(self, mock_store_result):
        """Test task completed hook."""
        # Setup mock to return a result ID
        mock_store_result.return_value = 123
        
        # Register with coordinator
        self.integration.register_with_coordinator()
        
        # Prepare test data
        task_id = "test_hook_task"
        worker_id = "test_hook_worker"
        
        # Create a task in the coordinator
        self.coordinator.tasks[task_id] = {
            "task_id": task_id,
            "type": "benchmark",
            "priority": 1,
            "requirements": {"hardware": ["cuda"]},
            "metadata": {"model": "test_model", "batch_size": 8}
        }
        
        # Prepare result
        result = {
            "status": "success",
            "metrics": {
                "throughput": 100.0,
                "latency": 10.0,
                "memory_usage": 1000.0
            }
        }
        
        # Call task completed hook
        await self.integration._hook_task_completed(task_id, worker_id, result, 10.5)
        
        # Verify store_result was called
        mock_store_result.assert_called_once()
        
        # Verify correct arguments
        args, _ = mock_store_result.call_args
        stored_data = args[0]
        
        # Check basic fields
        self.assertEqual(stored_data["task_id"], task_id)
        self.assertEqual(stored_data["worker_id"], worker_id)
        self.assertEqual(stored_data["type"], "benchmark")
        self.assertEqual(stored_data["status"], "completed")
        self.assertEqual(stored_data["duration"], 10.5)
        
        # Check metrics
        self.assertEqual(stored_data["metrics"]["throughput"], 100.0)
        self.assertEqual(stored_data["metrics"]["latency"], 10.0)
        self.assertEqual(stored_data["metrics"]["memory_usage"], 1000.0)
    
    @patch('result_aggregator.coordinator_integration.ResultAggregatorService.store_result')
    async def test_task_failed_hook(self, mock_store_result):
        """Test task failed hook."""
        # Setup mock to return a result ID
        mock_store_result.return_value = 123
        
        # Register with coordinator
        self.integration.register_with_coordinator()
        
        # Prepare test data
        task_id = "test_hook_task"
        worker_id = "test_hook_worker"
        
        # Create a task in the coordinator
        self.coordinator.tasks[task_id] = {
            "task_id": task_id,
            "type": "benchmark",
            "priority": 1,
            "requirements": {"hardware": ["cuda"]},
            "metadata": {"model": "test_model", "batch_size": 8},
            "attempts": 1
        }
        
        # Prepare error
        error = "RuntimeError: Out of memory"
        
        # Call task failed hook
        await self.integration._hook_task_failed(task_id, worker_id, error, 5.2)
        
        # Verify store_result was called
        mock_store_result.assert_called_once()
        
        # Verify correct arguments
        args, _ = mock_store_result.call_args
        stored_data = args[0]
        
        # Check basic fields
        self.assertEqual(stored_data["task_id"], task_id)
        self.assertEqual(stored_data["worker_id"], worker_id)
        self.assertEqual(stored_data["type"], "benchmark")
        self.assertEqual(stored_data["status"], "failed")
        self.assertEqual(stored_data["duration"], 5.2)
        
        # Check details
        self.assertEqual(stored_data["details"]["error"], error)
        self.assertEqual(stored_data["details"]["attempts"], 1)
        
        # Check metrics
        self.assertEqual(stored_data["metrics"]["execution_time"], 5.2)
        self.assertEqual(stored_data["metrics"]["error_occurred"], 1.0)
    
    @patch('result_aggregator.coordinator_integration.ResultAggregatorIntegration._send_notification')
    @patch('result_aggregator.service.ResultAggregatorService.detect_anomalies')
    @patch('result_aggregator.service.ResultAggregatorService.analyze_performance_trends')
    async def test_periodic_analysis(self, mock_analyze_trends, mock_detect_anomalies, mock_send_notification):
        """Test periodic analysis."""
        # Setup mocks
        mock_analyze_trends.return_value = {
            "throughput": {
                "trend": "increasing",
                "percent_change": 20.0,
                "statistics": {"mean": 100.0}
            },
            "latency": {
                "trend": "stable",
                "percent_change": 0.5,
                "statistics": {"mean": 10.0}
            }
        }
        
        mock_detect_anomalies.return_value = [
            {
                "score": 0.9,
                "type": "performance",
                "details": {
                    "task_type": "benchmark",
                    "anomalous_features": [
                        {"feature": "throughput", "value": 500.0, "z_score": 8.0}
                    ]
                }
            }
        ]
        
        # Initialize notification callbacks
        self.integration.notification_callbacks = [MagicMock()]
        
        # Call periodic analysis
        await self.integration._run_periodic_analysis()
        
        # Verify analyze_trends was called
        mock_analyze_trends.assert_called_once()
        
        # Verify detect_anomalies was called
        mock_detect_anomalies.assert_called_once()
        
        # Verify notifications were sent for both anomaly and trend
        self.assertEqual(mock_send_notification.call_count, 2)
        
        # Verify notification types
        call_args_list = mock_send_notification.call_args_list
        notification_types = [call[0][0]["type"] for call in call_args_list]
        self.assertIn("anomaly", notification_types)
        self.assertIn("trend", notification_types)


if __name__ == '__main__':
    unittest.main()