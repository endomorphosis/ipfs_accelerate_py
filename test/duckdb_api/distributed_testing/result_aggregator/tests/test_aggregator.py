#!/usr/bin/env python3
"""
Tests for the Result Aggregator module.
"""

import os
import sys
import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.duckdb.distributed_testing.result_aggregator import ResultAggregator

class TestResultAggregator(unittest.TestCase):
    """Test cases for ResultAggregator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock DB manager
        self.db_manager = MagicMock()
        
        # Create a mock task scheduler
        self.task_scheduler = MagicMock()
        
        # Configure the aggregator for testing
        config = {
            "update_interval": 0.1,  # Short interval for testing
            "visualization_enabled": False,  # Disable visualization for testing
            "database_enabled": True,
            "aggregate_dimensions": ["hardware", "model"],
            "comparison_metrics": ["throughput", "latency"],
            "regression_threshold": 0.05
        }
        
        # Create the aggregator
        self.aggregator = ResultAggregator(self.db_manager, self.task_scheduler)
        self.aggregator.configure(config)
        
        # Sample test results
        self.test_results = [
            {
                "test_id": "test1",
                "worker_id": "worker1",
                "task_type": "benchmark",
                "hardware_id": "gpu1",
                "model": "bert",
                "status": "success",
                "timestamp": datetime.now() - timedelta(days=1),
                "duration": 10.5,
                "throughput": 100.0,
                "latency": 50.0
            },
            {
                "test_id": "test1",
                "worker_id": "worker2",
                "task_type": "benchmark",
                "hardware_id": "gpu2",
                "model": "bert",
                "status": "success",
                "timestamp": datetime.now() - timedelta(hours=12),
                "duration": 9.8,
                "throughput": 110.0,
                "latency": 45.0
            },
            {
                "test_id": "test2",
                "worker_id": "worker1",
                "task_type": "inference",
                "hardware_id": "gpu1",
                "model": "t5",
                "status": "failed",
                "failure_reason": "timeout",
                "timestamp": datetime.now() - timedelta(hours=6),
                "duration": 60.0,
                "throughput": 20.0,
                "latency": 200.0
            },
            {
                "test_id": "test2",
                "worker_id": "worker2",
                "task_type": "inference",
                "hardware_id": "gpu2",
                "model": "t5",
                "status": "success",
                "timestamp": datetime.now() - timedelta(hours=3),
                "duration": 55.0,
                "throughput": 22.0,
                "latency": 180.0
            }
        ]
        
        # Configure DB manager mock
        self.db_manager.get_test_results.return_value = self.test_results
    
    def test_initialization(self):
        """Test initialization of the aggregator."""
        self.assertEqual(self.aggregator.db_manager, self.db_manager)
        self.assertEqual(self.aggregator.task_scheduler, self.task_scheduler)
        self.assertEqual(len(self.aggregator.test_results), 0)
        self.assertEqual(len(self.aggregator.worker_results), 0)
        self.assertEqual(len(self.aggregator.task_type_results), 0)
        self.assertEqual(len(self.aggregator.hardware_results), 0)
    
    def test_load_historical_data(self):
        """Test loading historical data."""
        self.aggregator._load_historical_data()
        
        # Verify that DB manager was called correctly
        self.db_manager.get_test_results.assert_called_once()
        
        # Verify that results were processed
        self.assertEqual(len(self.aggregator.test_results), 2)
        self.assertEqual(len(self.aggregator.worker_results), 2)
        self.assertEqual(len(self.aggregator.task_type_results), 2)
        self.assertEqual(len(self.aggregator.hardware_results), 2)
        
        # Verify test results
        self.assertEqual(len(self.aggregator.test_results["test1"]), 2)
        self.assertEqual(len(self.aggregator.test_results["test2"]), 2)
        
        # Verify worker results
        self.assertEqual(len(self.aggregator.worker_results["worker1"]), 2)
        self.assertEqual(len(self.aggregator.worker_results["worker2"]), 2)
        
        # Verify task type results
        self.assertEqual(len(self.aggregator.task_type_results["benchmark"]), 2)
        self.assertEqual(len(self.aggregator.task_type_results["inference"]), 2)
        
        # Verify hardware results
        self.assertEqual(len(self.aggregator.hardware_results["gpu1"]), 2)
        self.assertEqual(len(self.aggregator.hardware_results["gpu2"]), 2)
    
    def test_aggregate_results(self):
        """Test aggregating results."""
        # Load historical data first
        self.aggregator._load_historical_data()
        
        # Aggregate results
        self.aggregator._aggregate_results()
        
        # Verify dimension analysis
        self.assertIn("hardware", self.aggregator.dimension_analysis)
        self.assertIn("model", self.aggregator.dimension_analysis)
        
        # Verify hardware dimension
        self.assertIn("gpu1", self.aggregator.dimension_analysis["hardware"])
        self.assertIn("gpu2", self.aggregator.dimension_analysis["hardware"])
        
        # Verify model dimension
        self.assertIn("bert", self.aggregator.dimension_analysis["model"])
        self.assertIn("t5", self.aggregator.dimension_analysis["model"])
        
        # Verify aggregated metrics
        self.assertIn("duration_mean", self.aggregator.dimension_analysis["model"]["bert"])
        self.assertIn("throughput_mean", self.aggregator.dimension_analysis["model"]["bert"])
        self.assertIn("latency_mean", self.aggregator.dimension_analysis["model"]["bert"])
    
    def test_analyze_results(self):
        """Test analyzing results."""
        # Load and aggregate data first
        self.aggregator._load_historical_data()
        self.aggregator._aggregate_results()
        
        # Analyze results
        self.aggregator._analyze_results()
        
        # Verify test analysis
        self.assertIn("test1", self.aggregator.test_analysis)
        self.assertIn("test2", self.aggregator.test_analysis)
        
        # Verify worker analysis
        self.assertIn("worker1", self.aggregator.worker_analysis)
        self.assertIn("worker2", self.aggregator.worker_analysis)
        
        # Verify task type analysis
        self.assertIn("benchmark", self.aggregator.task_type_analysis)
        self.assertIn("inference", self.aggregator.task_type_analysis)
        
        # Verify hardware analysis
        self.assertIn("gpu1", self.aggregator.hardware_analysis)
        self.assertIn("gpu2", self.aggregator.hardware_analysis)
    
    def test_start_stop(self):
        """Test starting and stopping the aggregator."""
        # Start the aggregator
        with patch.object(self.aggregator, '_load_historical_data') as mock_load:
            with patch.object(self.aggregator, '_aggregate_results') as mock_aggregate:
                with patch.object(self.aggregator, '_analyze_results') as mock_analyze:
                    self.aggregator.start()
                    
                    # Verify that methods were called
                    mock_load.assert_called_once()
                    mock_aggregate.assert_called_once()
                    mock_analyze.assert_called_once()
                    
                    # Verify that update thread was started
                    self.assertTrue(self.aggregator.update_thread.is_alive())
                    
                    # Stop the aggregator
                    self.aggregator.stop()
                    
                    # Verify that update thread was stopped
                    self.assertFalse(self.aggregator.update_thread.is_alive())
    
    def test_get_overall_status(self):
        """Test getting overall status."""
        # Load and analyze data first
        self.aggregator._load_historical_data()
        self.aggregator._aggregate_results()
        self.aggregator._analyze_results()
        
        # Get overall status
        status = self.aggregator.get_overall_status()
        
        # Verify status
        self.assertEqual(status["test_count"], 2)
        self.assertEqual(status["worker_count"], 2)
        self.assertEqual(status["task_type_count"], 2)
        self.assertEqual(status["hardware_count"], 2)
        self.assertEqual(status["total_executions"], 4)
    
    def test_generate_report(self):
        """Test generating reports."""
        # Load and analyze data first
        self.aggregator._load_historical_data()
        self.aggregator._aggregate_results()
        self.aggregator._analyze_results()
        
        # Test JSON report
        json_report = self.aggregator.generate_report(format="json")
        self.assertIsNotNone(json_report)
        parsed = json.loads(json_report)
        self.assertIn("overall_status", parsed)
        self.assertIn("test_analysis", parsed)
        
        # Test markdown report
        md_report = self.aggregator.generate_report(format="md")
        self.assertIsNotNone(md_report)
        self.assertIn("# Distributed Testing Results Report", md_report)
        
        # Test HTML report
        html_report = self.aggregator.generate_report(format="html")
        self.assertIsNotNone(html_report)
        self.assertIn("<html>", html_report)

if __name__ == "__main__":
    unittest.main()