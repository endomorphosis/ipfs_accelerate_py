#!/usr/bin/env python3
"""
Tests for the Performance Trend Analyzer of the Distributed Testing Framework.

This module tests the performance trend analysis capabilities of the framework,
including time series tracking, trend analysis, and anomaly detection.
"""

import os
import sys
import unittest
import tempfile
import threading
import time
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from duckdb_api.distributed_testing.performance_trend_analyzer import PerformanceTrendAnalyzer

class TestPerformanceTrendAnalyzer(unittest.TestCase):
    """Test cases for the Performance Trend Analyzer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = MagicMock()
        self.task_scheduler = MagicMock()
        
        # Create test performance trend analyzer
        self.analyzer = PerformanceTrendAnalyzer(
            db_manager=self.db_manager,
            task_scheduler=self.task_scheduler
        )
        
        # Configure for testing
        self.analyzer.configure({
            "history_days": 30,
            "anomaly_threshold": 3.0,
            "trend_significance_threshold": 0.05,
            "forecast_days": 7,
            "min_data_points": 5,
            "update_interval": 0.1,  # 100ms for testing
            "visualization_enabled": True,
            "visualization_format": "png",
            "visualization_path": os.path.join(self.temp_dir, "visualizations"),
            "database_enabled": True,
            "metrics": ["execution_time", "success_rate", "throughput", "memory_usage", "cpu_usage"],
            "aggregate_metrics": True
        })
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop analyzer if running
        if hasattr(self, 'analyzer'):
            self.analyzer.stop()
            
        # Remove temp directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_initialization(self):
        """Test initialization of PerformanceTrendAnalyzer."""
        # Check initial state
        self.assertEqual(len(self.analyzer.time_series), 0)
        self.assertEqual(len(self.analyzer.task_time_series), 0)
        self.assertEqual(len(self.analyzer.worker_baselines), 0)
        self.assertEqual(len(self.analyzer.task_baselines), 0)
        self.assertEqual(len(self.analyzer.anomalies), 0)
        self.assertEqual(len(self.analyzer.task_anomalies), 0)
        self.assertEqual(len(self.analyzer.trends), 0)
        self.assertEqual(len(self.analyzer.task_trends), 0)
        
    def test_update_time_series(self):
        """Test updating time series data."""
        # Mock task scheduler worker performance data
        self.task_scheduler.worker_performance = {
            "worker1": {
                "success_count": 80,
                "task_count": 100,
                "total_execution_time": 500,
                "cpu_percent": 50,
                "available_memory_gb": 16,
                "task_types": {
                    "benchmark": {
                        "success_rate": 0.8,
                        "avg_execution_time": 5.0,
                        "task_count": 100
                    }
                }
            }
        }
        
        # Mock task scheduler task stats
        self.task_scheduler.task_stats = {
            "benchmark": {
                "success_rate": 0.8,
                "avg_execution_time": 5.0,
                "task_count": 100
            }
        }
        
        # Update time series
        self.analyzer._update_time_series()
        
        # Check worker time series
        self.assertIn("worker1", self.analyzer.time_series)
        self.assertIn("success_rate", self.analyzer.time_series["worker1"])
        self.assertIn("avg_execution_time", self.analyzer.time_series["worker1"])
        self.assertIn("cpu_percent", self.analyzer.time_series["worker1"])
        self.assertIn("available_memory_gb", self.analyzer.time_series["worker1"])
        
        # Check task time series
        self.assertIn("benchmark", self.analyzer.task_time_series)
        self.assertIn("success_rate", self.analyzer.task_time_series["benchmark"])
        self.assertIn("avg_execution_time", self.analyzer.task_time_series["benchmark"])
        self.assertIn("task_count", self.analyzer.task_time_series["benchmark"])
        
        # Check worker time series data
        success_rate_data = self.analyzer.time_series["worker1"]["success_rate"]
        self.assertEqual(len(success_rate_data), 1)
        timestamp, value = success_rate_data[0]
        self.assertIsInstance(timestamp, datetime)
        self.assertEqual(value, 0.8)
        
        # Check task time series data
        success_rate_data = self.analyzer.task_time_series["benchmark"]["success_rate"]
        self.assertEqual(len(success_rate_data), 1)
        timestamp, value = success_rate_data[0]
        self.assertIsInstance(timestamp, datetime)
        self.assertEqual(value, 0.8)
        
        # Check database integration
        self.db_manager.add_worker_performance_metrics.assert_called_once()
        self.db_manager.add_task_type_performance_metrics.assert_called_once()
        
    def test_prune_time_series(self):
        """Test pruning old time series data."""
        # Add some test data
        current_time = datetime.now()
        old_time = current_time - timedelta(days=31)  # Older than history_days
        
        # Worker time series
        self.analyzer.time_series["worker1"] = {
            "success_rate": [
                (current_time, 0.8),
                (old_time, 0.7)  # Old data to be pruned
            ],
            "avg_execution_time": [
                (current_time, 5.0)
            ]
        }
        
        # Task time series
        self.analyzer.task_time_series["benchmark"] = {
            "success_rate": [
                (current_time, 0.8),
                (old_time, 0.7)  # Old data to be pruned
            ]
        }
        
        # Prune time series
        self.analyzer._prune_time_series()
        
        # Check worker time series
        self.assertEqual(len(self.analyzer.time_series["worker1"]["success_rate"]), 1)
        self.assertEqual(self.analyzer.time_series["worker1"]["success_rate"][0][0], current_time)
        self.assertEqual(self.analyzer.time_series["worker1"]["success_rate"][0][1], 0.8)
        
        # Check task time series
        self.assertEqual(len(self.analyzer.task_time_series["benchmark"]["success_rate"]), 1)
        self.assertEqual(self.analyzer.task_time_series["benchmark"]["success_rate"][0][0], current_time)
        self.assertEqual(self.analyzer.task_time_series["benchmark"]["success_rate"][0][1], 0.8)
        
    def test_compute_baselines(self):
        """Test computing performance baselines."""
        # Add some test data
        current_time = datetime.now()
        
        # Worker time series with enough data points
        self.analyzer.time_series["worker1"] = {
            "success_rate": [
                (current_time - timedelta(minutes=5*i), 0.8) for i in range(6)
            ],
            "avg_execution_time": [
                (current_time - timedelta(minutes=5*i), 5.0) for i in range(6)
            ]
        }
        
        # Task time series with enough data points
        self.analyzer.task_time_series["benchmark"] = {
            "success_rate": [
                (current_time - timedelta(minutes=5*i), 0.8) for i in range(6)
            ],
            "avg_execution_time": [
                (current_time - timedelta(minutes=5*i), 5.0) for i in range(6)
            ]
        }
        
        # Worker time series with not enough data points
        self.analyzer.time_series["worker2"] = {
            "success_rate": [
                (current_time, 0.9)
            ]
        }
        
        # Compute baselines
        self.analyzer._compute_baselines()
        
        # Check worker baselines
        self.assertIn("worker1", self.analyzer.worker_baselines)
        self.assertIn("success_rate", self.analyzer.worker_baselines["worker1"])
        self.assertIn("avg_execution_time", self.analyzer.worker_baselines["worker1"])
        
        # Check task baselines
        self.assertIn("benchmark", self.analyzer.task_baselines)
        self.assertIn("success_rate", self.analyzer.task_baselines["benchmark"])
        self.assertIn("avg_execution_time", self.analyzer.task_baselines["benchmark"])
        
        # Check baseline values
        self.assertEqual(self.analyzer.worker_baselines["worker1"]["success_rate"]["mean"], 0.8)
        self.assertEqual(self.analyzer.worker_baselines["worker1"]["avg_execution_time"]["mean"], 5.0)
        
        # Check worker2 (not enough data points)
        self.assertNotIn("worker2", self.analyzer.worker_baselines)
        
    def test_detect_anomalies(self):
        """Test detecting anomalies in performance metrics."""
        # Add some test data
        current_time = datetime.now()
        
        # Create baselines
        self.analyzer.worker_baselines["worker1"] = {
            "success_rate": {
                "mean": 0.8,
                "stdev": 0.05,
                "threshold": 3.0,
                "updated_at": current_time,
                "n_samples": 10
            },
            "avg_execution_time": {
                "mean": 5.0,
                "stdev": 0.5,
                "threshold": 3.0,
                "updated_at": current_time,
                "n_samples": 10
            }
        }
        
        self.analyzer.task_baselines["benchmark"] = {
            "success_rate": {
                "mean": 0.8,
                "stdev": 0.05,
                "threshold": 3.0,
                "updated_at": current_time,
                "n_samples": 10
            }
        }
        
        # Add time series data with anomalies
        self.analyzer.time_series["worker1"] = {
            "success_rate": [
                (current_time - timedelta(minutes=5*i), 0.8) for i in range(5)
            ] + [
                (current_time, 0.5)  # Anomaly: 6 std devs below mean
            ],
            "avg_execution_time": [
                (current_time - timedelta(minutes=5*i), 5.0) for i in range(5)
            ] + [
                (current_time, 10.0)  # Anomaly: 10 std devs above mean
            ]
        }
        
        self.analyzer.task_time_series["benchmark"] = {
            "success_rate": [
                (current_time - timedelta(minutes=5*i), 0.8) for i in range(5)
            ] + [
                (current_time, 0.5)  # Anomaly: 6 std devs below mean
            ]
        }
        
        # Detect anomalies
        self.analyzer._detect_anomalies()
        
        # Check worker anomalies
        self.assertIn("worker1", self.analyzer.anomalies)
        self.assertEqual(len(self.analyzer.anomalies["worker1"]), 2)
        
        # Check anomaly details for success_rate
        anomaly = next(a for a in self.analyzer.anomalies["worker1"] if a["metric"] == "success_rate")
        self.assertEqual(anomaly["value"], 0.5)
        self.assertEqual(anomaly["mean"], 0.8)
        self.assertEqual(anomaly["stdev"], 0.05)
        self.assertLess(anomaly["z_score"], 0)  # Negative z-score (below mean)
        self.assertFalse(anomaly["is_high"])
        
        # Check anomaly details for avg_execution_time
        anomaly = next(a for a in self.analyzer.anomalies["worker1"] if a["metric"] == "avg_execution_time")
        self.assertEqual(anomaly["value"], 10.0)
        self.assertEqual(anomaly["mean"], 5.0)
        self.assertEqual(anomaly["stdev"], 0.5)
        self.assertGreater(anomaly["z_score"], 0)  # Positive z-score (above mean)
        self.assertTrue(anomaly["is_high"])
        
        # Check task anomalies
        self.assertIn("benchmark", self.analyzer.task_anomalies)
        self.assertEqual(len(self.analyzer.task_anomalies["benchmark"]), 1)
        
        # Check anomaly details for success_rate
        anomaly = self.analyzer.task_anomalies["benchmark"][0]
        self.assertEqual(anomaly["metric"], "success_rate")
        self.assertEqual(anomaly["value"], 0.5)
        self.assertEqual(anomaly["mean"], 0.8)
        self.assertEqual(anomaly["stdev"], 0.05)
        self.assertLess(anomaly["z_score"], 0)  # Negative z-score (below mean)
        self.assertFalse(anomaly["is_high"])
        
        # Check database integration
        self.db_manager.add_performance_anomaly.assert_called()
        
    def test_analyze_trends(self):
        """Test analyzing trends in performance metrics."""
        # Add some test data with clear trend
        current_time = datetime.now()
        
        # Worker time series with a clear trend
        self.analyzer.time_series["worker1"] = {
            "success_rate": [
                (current_time - timedelta(days=i), 0.8 - 0.02 * i) for i in range(10)
            ],  # Decreasing trend
            "avg_execution_time": [
                (current_time - timedelta(days=i), 5.0 + 0.5 * i) for i in range(10)
            ]  # Increasing trend
        }
        
        # Task time series with a clear trend
        self.analyzer.task_time_series["benchmark"] = {
            "success_rate": [
                (current_time - timedelta(days=i), 0.8 - 0.02 * i) for i in range(10)
            ]  # Decreasing trend
        }
        
        # Analyze trends
        self.analyzer._analyze_trends()
        
        # Check worker trends
        self.assertIn("worker1", self.analyzer.trends)
        self.assertIn("success_rate", self.analyzer.trends["worker1"])
        self.assertIn("avg_execution_time", self.analyzer.trends["worker1"])
        
        # Check success_rate trend (should be decreasing)
        trend = self.analyzer.trends["worker1"]["success_rate"]
        self.assertLess(trend["slope"], 0)  # Negative slope
        self.assertLess(trend["p_value"], 0.05)  # Significant p-value
        self.assertTrue(trend["is_significant"])
        self.assertEqual(trend["direction"], "decreasing")
        self.assertEqual(len(trend["forecast"]), 7)  # 7 forecast days
        
        # Check avg_execution_time trend (should be increasing)
        trend = self.analyzer.trends["worker1"]["avg_execution_time"]
        self.assertGreater(trend["slope"], 0)  # Positive slope
        self.assertLess(trend["p_value"], 0.05)  # Significant p-value
        self.assertTrue(trend["is_significant"])
        self.assertEqual(trend["direction"], "increasing")
        self.assertEqual(len(trend["forecast"]), 7)  # 7 forecast days
        
        # Check task trends
        self.assertIn("benchmark", self.analyzer.task_trends)
        self.assertIn("success_rate", self.analyzer.task_trends["benchmark"])
        
        # Check success_rate trend (should be decreasing)
        trend = self.analyzer.task_trends["benchmark"]["success_rate"]
        self.assertLess(trend["slope"], 0)  # Negative slope
        self.assertLess(trend["p_value"], 0.05)  # Significant p-value
        self.assertTrue(trend["is_significant"])
        self.assertEqual(trend["direction"], "decreasing")
        self.assertEqual(len(trend["forecast"]), 7)  # 7 forecast days
        
        # Check database integration
        self.db_manager.add_performance_trend.assert_called()
        
    def test_get_worker_trends(self):
        """Test getting worker trends."""
        # Add some test trends
        self.analyzer.trends = {
            "worker1": {
                "success_rate": {
                    "slope": -0.02,
                    "p_value": 0.01,
                    "r_squared": 0.9,
                    "is_significant": True,
                    "forecast": [(1, 0.78), (2, 0.76)],
                    "direction": "decreasing",
                    "updated_at": datetime.now()
                },
                "avg_execution_time": {
                    "slope": 0.5,
                    "p_value": 0.01,
                    "r_squared": 0.9,
                    "is_significant": True,
                    "forecast": [(1, 5.5), (2, 6.0)],
                    "direction": "increasing",
                    "updated_at": datetime.now()
                }
            },
            "worker2": {
                "success_rate": {
                    "slope": -0.01,
                    "p_value": 0.1,  # Not significant
                    "r_squared": 0.3,
                    "is_significant": False,
                    "forecast": [(1, 0.79), (2, 0.78)],
                    "direction": "decreasing",
                    "updated_at": datetime.now()
                }
            }
        }
        
        # Get all worker trends
        trends = self.analyzer.get_worker_trends()
        self.assertEqual(len(trends), 2)
        self.assertIn("worker1", trends)
        self.assertIn("worker2", trends)
        
        # Get specific worker trends
        trends = self.analyzer.get_worker_trends("worker1")
        self.assertEqual(len(trends), 2)
        self.assertIn("success_rate", trends)
        self.assertIn("avg_execution_time", trends)
        
        # Get specific metric for specific worker
        trends = self.analyzer.get_worker_trends("worker1", "success_rate")
        self.assertEqual(len(trends), 1)
        self.assertIn("success_rate", trends)
        
        # Get significant trends only
        trends = self.analyzer.get_worker_trends(significant_only=True)
        self.assertEqual(len(trends), 1)
        self.assertIn("worker1", trends)
        self.assertNotIn("worker2", trends)
        
    def test_get_task_trends(self):
        """Test getting task trends."""
        # Add some test trends
        self.analyzer.task_trends = {
            "benchmark": {
                "success_rate": {
                    "slope": -0.02,
                    "p_value": 0.01,
                    "r_squared": 0.9,
                    "is_significant": True,
                    "forecast": [(1, 0.78), (2, 0.76)],
                    "direction": "decreasing",
                    "updated_at": datetime.now()
                }
            },
            "test": {
                "success_rate": {
                    "slope": -0.01,
                    "p_value": 0.1,  # Not significant
                    "r_squared": 0.3,
                    "is_significant": False,
                    "forecast": [(1, 0.79), (2, 0.78)],
                    "direction": "decreasing",
                    "updated_at": datetime.now()
                }
            }
        }
        
        # Get all task trends
        trends = self.analyzer.get_task_trends()
        self.assertEqual(len(trends), 2)
        self.assertIn("benchmark", trends)
        self.assertIn("test", trends)
        
        # Get specific task trends
        trends = self.analyzer.get_task_trends("benchmark")
        self.assertEqual(len(trends), 1)
        self.assertIn("success_rate", trends)
        
        # Get specific metric for specific task
        trends = self.analyzer.get_task_trends("benchmark", "success_rate")
        self.assertEqual(len(trends), 1)
        self.assertIn("success_rate", trends)
        
        # Get significant trends only
        trends = self.analyzer.get_task_trends(significant_only=True)
        self.assertEqual(len(trends), 1)
        self.assertIn("benchmark", trends)
        self.assertNotIn("test", trends)
        
    def test_get_worker_anomalies(self):
        """Test getting worker anomalies."""
        # Add some test anomalies
        current_time = datetime.now()
        
        self.analyzer.anomalies = {
            "worker1": [
                {
                    "worker_id": "worker1",
                    "metric": "success_rate",
                    "timestamp": current_time,
                    "value": 0.5,
                    "mean": 0.8,
                    "stdev": 0.05,
                    "z_score": -6.0,
                    "is_high": False,
                    "detected_at": current_time
                },
                {
                    "worker_id": "worker1",
                    "metric": "avg_execution_time",
                    "timestamp": current_time,
                    "value": 10.0,
                    "mean": 5.0,
                    "stdev": 0.5,
                    "z_score": 10.0,
                    "is_high": True,
                    "detected_at": current_time
                }
            ],
            "worker2": [
                {
                    "worker_id": "worker2",
                    "metric": "success_rate",
                    "timestamp": current_time,
                    "value": 0.5,
                    "mean": 0.8,
                    "stdev": 0.05,
                    "z_score": -6.0,
                    "is_high": False,
                    "detected_at": current_time
                }
            ]
        }
        
        # Get all worker anomalies
        anomalies = self.analyzer.get_worker_anomalies()
        self.assertEqual(len(anomalies), 2)
        self.assertIn("worker1", anomalies)
        self.assertIn("worker2", anomalies)
        
        # Get specific worker anomalies
        anomalies = self.analyzer.get_worker_anomalies("worker1")
        self.assertEqual(len(anomalies), 2)
        self.assertIn("success_rate", anomalies)
        self.assertIn("avg_execution_time", anomalies)
        
        # Get specific metric for specific worker
        anomalies = self.analyzer.get_worker_anomalies("worker1", "success_rate")
        self.assertEqual(len(anomalies), 1)
        self.assertIn("success_rate", anomalies)
        self.assertEqual(len(anomalies["success_rate"]), 1)
        
        # Test limit
        anomalies = self.analyzer.get_worker_anomalies("worker1", limit=1)
        self.assertEqual(len(anomalies), 2)  # Still two metrics
        self.assertEqual(len(anomalies["success_rate"]), 1)  # But only one anomaly per metric
        
    def test_generate_visualization(self):
        """Test generating visualizations."""
        # Add some test data
        current_time = datetime.now()
        
        # Create baselines
        self.analyzer.worker_baselines["worker1"] = {
            "success_rate": {
                "mean": 0.8,
                "stdev": 0.05,
                "threshold": 3.0,
                "updated_at": current_time,
                "n_samples": 10
            }
        }
        
        # Add time series data
        self.analyzer.time_series["worker1"] = {
            "success_rate": [
                (current_time - timedelta(days=i), 0.8 - 0.02 * i) for i in range(10)
            ]  # Decreasing trend
        }
        
        # Add trend analysis
        self.analyzer.trends["worker1"] = {
            "success_rate": {
                "slope": -0.02,
                "p_value": 0.01,
                "r_squared": 0.9,
                "is_significant": True,
                "forecast": [(1, 0.78), (2, 0.76), (3, 0.74), (4, 0.72), (5, 0.70), (6, 0.68), (7, 0.66)],
                "direction": "decreasing",
                "updated_at": current_time
            }
        }
        
        # Add anomaly
        self.analyzer.anomalies["worker1"] = [
            {
                "worker_id": "worker1",
                "metric": "success_rate",
                "timestamp": current_time,
                "value": 0.5,
                "mean": 0.8,
                "stdev": 0.05,
                "z_score": -6.0,
                "is_high": False,
                "detected_at": current_time
            }
        ]
        
        # Generate visualization
        filepath = self.analyzer.generate_visualization("worker1", "worker", "success_rate")
        
        # Check file was created
        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))
        
    def test_get_performance_report(self):
        """Test generating performance reports."""
        # Add some test data
        current_time = datetime.now()
        
        # Add worker trends
        self.analyzer.trends = {
            "worker1": {
                "success_rate": {
                    "slope": -0.02,
                    "p_value": 0.01,
                    "r_squared": 0.9,
                    "is_significant": True,
                    "forecast": [(1, 0.78), (2, 0.76)],
                    "direction": "decreasing",
                    "updated_at": current_time
                }
            }
        }
        
        # Add worker anomalies
        self.analyzer.anomalies = {
            "worker1": [
                {
                    "worker_id": "worker1",
                    "metric": "success_rate",
                    "timestamp": current_time,
                    "value": 0.5,
                    "mean": 0.8,
                    "stdev": 0.05,
                    "z_score": -6.0,
                    "is_high": False,
                    "detected_at": current_time
                }
            ]
        }
        
        # Add worker baselines
        self.analyzer.worker_baselines = {
            "worker1": {
                "success_rate": {
                    "mean": 0.8,
                    "stdev": 0.05,
                    "threshold": 3.0,
                    "updated_at": current_time,
                    "n_samples": 10
                }
            }
        }
        
        # Generate worker report
        report = self.analyzer.get_performance_report("worker", significant_only=True)
        
        # Check report structure
        self.assertIn("timestamp", report)
        self.assertIn("trends", report)
        self.assertIn("anomalies", report)
        self.assertIn("baselines", report)
        self.assertEqual(report["entity_type"], "worker")
        
        # Check report content
        self.assertIn("worker1", report["trends"])
        self.assertIn("success_rate", report["trends"]["worker1"])
        self.assertIn("worker1", report["anomalies"])
        self.assertIn("success_rate", report["anomalies"]["worker1"])
        self.assertIn("worker1", report["baselines"])
        self.assertIn("success_rate", report["baselines"]["worker1"])
        
        # Generate task report
        report = self.analyzer.get_performance_report("task_type", significant_only=True)
        
        # Check report structure
        self.assertIn("timestamp", report)
        self.assertIn("trends", report)
        self.assertIn("anomalies", report)
        self.assertIn("baselines", report)
        self.assertEqual(report["entity_type"], "task_type")
        
    def test_export_data(self):
        """Test exporting performance data."""
        # Add some test data
        current_time = datetime.now()
        
        # Add time series data
        self.analyzer.time_series["worker1"] = {
            "success_rate": [
                (current_time - timedelta(days=i), 0.8 - 0.02 * i) for i in range(3)
            ]
        }
        
        # Add worker trends
        self.analyzer.trends = {
            "worker1": {
                "success_rate": {
                    "slope": -0.02,
                    "p_value": 0.01,
                    "r_squared": 0.9,
                    "is_significant": True,
                    "forecast": [(1, 0.78), (2, 0.76)],
                    "direction": "decreasing",
                    "updated_at": current_time
                }
            }
        }
        
        # Create export path
        export_path = os.path.join(self.temp_dir, "export_data.json")
        
        # Export data to JSON
        success = self.analyzer.export_data(export_path, "json")
        
        # Check export
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Check file content
        with open(export_path, 'r') as f:
            data = json.load(f)
            
        # Check data structure
        self.assertIn("time_series", data)
        self.assertIn("trends", data)
        self.assertIn("export_timestamp", data)
        
        # Check time series data
        self.assertIn("worker1", data["time_series"])
        self.assertIn("success_rate", data["time_series"]["worker1"])
        self.assertEqual(len(data["time_series"]["worker1"]["success_rate"]), 3)
        
        # Check trend data
        self.assertIn("worker1", data["trends"])
        self.assertIn("success_rate", data["trends"]["worker1"])
        self.assertEqual(data["trends"]["worker1"]["success_rate"]["slope"], -0.02)
        
        # Export data to CSV
        export_path = os.path.join(self.temp_dir, "export_data.csv")
        success = self.analyzer.export_data(export_path, "csv")
        
        # Check export
        self.assertTrue(success)
        self.assertTrue(os.path.exists(os.path.splitext(export_path)[0] + "_worker_timeseries.csv"))
        
if __name__ == '__main__':
    unittest.main()