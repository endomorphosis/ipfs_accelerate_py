#!/usr/bin/env python3
"""
Integration test for the Performance Trend Analyzer with Coordinator.

This test verifies that the Performance Trend Analyzer properly collects and
analyzes metrics from the coordinator, detects anomalies and trends, and
generates appropriate reports and visualizations.
"""

import anyio
import json
import os
import sys
import time
import unittest
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add /test to sys.path so that `distributed_testing` resolves to `test/distributed_testing`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .integration_mode import integration_enabled, integration_opt_in_message

if not integration_enabled():
    pytest.skip(integration_opt_in_message(), allow_module_level=True)

pytest.importorskip("aiohttp")

# Import the components to test
from .performance_trend_analyzer import (
    PerformanceTrendAnalyzer,
    PerformanceMetric,
    PerformanceAlert,
    PerformanceTrend
)
from .coordinator import TestCoordinator


class TestPerformanceTrendAnalyzerIntegration(unittest.TestCase):
    """Test integration between Performance Trend Analyzer and Coordinator."""
    
    def setUp(self):
        """Set up the test environment."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.test_dir.name, "reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a DuckDB path
        self.db_path = os.path.join(self.test_dir.name, "performance_metrics.db")
        
        # Create analyzer configuration
        self.config_path = os.path.join(self.test_dir.name, "analyzer_config.json")
        with open(self.config_path, "w") as f:
            json.dump({
                "polling_interval": 1,  # 1 second for fast testing
                "history_window": 30,  # 30 seconds for testing
                "anomaly_detection": {
                    "z_score_threshold": 2.0,  # Lower threshold for testing
                    "isolation_forest": {
                        "enabled": True,
                        "contamination": 0.1  # Higher contamination for testing
                    },
                    "moving_average": {
                        "enabled": True,
                        "window_size": 3,  # Smaller window for testing
                        "threshold_factor": 1.5  # Lower threshold for testing
                    }
                },
                "trend_analysis": {
                    "minimum_data_points": 5,  # Fewer points for testing
                    "regression_confidence_threshold": 0.6  # Lower confidence for testing
                },
                "reporting": {
                    "generate_charts": True,
                    "alert_thresholds": {
                        "latency_ms": {"warning": 1.3, "critical": 1.8},
                        "throughput_items_per_second": {"warning": 0.8, "critical": 0.6}
                    },
                    "email_alerts": False
                }
            }, f)
        
        # Start the coordinator
        self.coordinator_port = 8766  # Use a non-standard port to avoid conflicts
        self.coordinator = TestCoordinator(
            host='localhost',
            port=self.coordinator_port,
            heartbeat_interval=1,
            worker_timeout=5,
            high_availability=False
        )
        self.coordinator.start()
        
        # Allow coordinator to initialize
        time.sleep(1)
        
        # Set up the performance trend analyzer
        self.analyzer = PerformanceTrendAnalyzer(
            coordinator_url=f"http://localhost:{self.coordinator_port}",
            db_path=self.db_path,
            config_path=self.config_path,
            output_dir=self.output_dir
        )
    
    def tearDown(self):
        """Clean up the test environment."""
        # Stop the coordinator
        self.coordinator.stop()
        
        # Clean up temporary directory
        self.test_dir.cleanup()
    
    async def async_setUp(self):
        """Set up async components."""
        # Start the analyzer
        await self.analyzer.start()
    
    async def async_tearDown(self):
        """Clean up async components."""
        # Stop the analyzer
        await self.analyzer.stop()

    def test_analyzer_connects_to_coordinator(self):
        anyio.run(self._test_analyzer_connects_to_coordinator)

    async def _test_analyzer_connects_to_coordinator(self):
        """Test that the analyzer successfully connects to the coordinator."""
        # Set up
        try:
            # Manually connect (without starting the full analyzer)
            connected = await self.analyzer.connect()
            
            # Check that the connection was successful
            self.assertTrue(connected)
            self.assertIsNotNone(self.analyzer.session)
        finally:
            # Clean up
            if self.analyzer.session:
                await self.analyzer.session.close()

    def test_analyzer_collects_metrics(self):
        anyio.run(self._test_analyzer_collects_metrics)

    async def _test_analyzer_collects_metrics(self):
        """Test that the analyzer successfully collects metrics from the coordinator."""
        # Set up
        await self.async_setUp()
        
        try:
            # Mock the coordinator responses
            original_get = self.analyzer.session.get
            
            async def mock_get(url, *args, **kwargs):
                if "/task_results" in url:
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={
                        "results": [
                            {
                                "task_id": "task1",
                                "worker_id": "worker1",
                                "end_time": time.time(),
                                "type": "benchmark",
                                "hardware_metrics": {
                                    "cpu_percent": 50.0,
                                    "memory_percent": 60.0
                                },
                                "result": {
                                    "model": "bert-base-uncased",
                                    "batch_sizes": {
                                        "1": {"latency_ms": 100.0, "throughput_items_per_second": 10.0},
                                        "8": {"latency_ms": 250.0, "throughput_items_per_second": 32.0}
                                    },
                                    "precision": "fp32",
                                    "iterations": 100
                                },
                                "execution_time_seconds": 120.0
                            },
                            {
                                "task_id": "task2",
                                "worker_id": "worker2",
                                "end_time": time.time(),
                                "type": "test",
                                "hardware_metrics": {
                                    "cpu_percent": 40.0,
                                    "memory_percent": 50.0
                                },
                                "result": {
                                    "model": "vit-base-patch16-224",
                                    "duration_seconds": 45.0,
                                    "test_file": "test_vit.py",
                                    "test_count": 10,
                                    "passed": 10,
                                    "failed": 0
                                },
                                "execution_time_seconds": 50.0
                            }
                        ]
                    })
                    return mock_response
                elif "/system_metrics" in url:
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={
                        "workers": [
                            {
                                "id": "worker1",
                                "hardware_metrics": {
                                    "cpu_percent": 50.0,
                                    "memory_percent": 60.0,
                                    "memory_used_gb": 4.5,
                                    "gpu": [
                                        {"memory_utilization_percent": 75.0}
                                    ]
                                }
                            },
                            {
                                "id": "worker2",
                                "hardware_metrics": {
                                    "cpu_percent": 40.0,
                                    "memory_percent": 50.0,
                                    "memory_used_gb": 3.8
                                }
                            }
                        ],
                        "coordinator": {
                            "task_processing_rate": 2.5,
                            "avg_task_duration": 85.0,
                            "queue_length": 3
                        }
                    })
                    return mock_response
                else:
                    return await original_get(url, *args, **kwargs)
            
            # Apply the mock
            with patch.object(self.analyzer.session, 'get', side_effect=mock_get):
                # Wait for metric collection
                await anyio.sleep(2)
                
                # Check that metrics were collected
                self.assertGreater(len(self.analyzer.metrics_cache), 0)
                
                # Check for specific metrics
                latency_metrics = self.analyzer.metrics_cache.get("latency_ms", [])
                throughput_metrics = self.analyzer.metrics_cache.get("throughput_items_per_second", [])
                cpu_metrics = self.analyzer.metrics_cache.get("cpu_percent", [])
                
                self.assertGreater(len(latency_metrics), 0)
                self.assertGreater(len(throughput_metrics), 0)
                self.assertGreater(len(cpu_metrics), 0)
        finally:
            await self.async_tearDown()

    def test_analyzer_detects_anomalies(self):
        anyio.run(self._test_analyzer_detects_anomalies)

    async def _test_analyzer_detects_anomalies(self):
        """Test that the analyzer successfully detects anomalies in metrics."""
        # Set up
        await self.async_setUp()
        
        try:
            # Inject a series of normal metrics
            timestamps = [time.time() - i for i in range(10, 0, -1)]  # Recent to old
            
            # Create metrics for throughput with an anomaly
            throughput_metrics = [
                PerformanceMetric(name="throughput_items_per_second", value=100.0, timestamp=timestamps[0]),
                PerformanceMetric(name="throughput_items_per_second", value=105.0, timestamp=timestamps[1]),
                PerformanceMetric(name="throughput_items_per_second", value=98.0, timestamp=timestamps[2]),
                PerformanceMetric(name="throughput_items_per_second", value=30.0, timestamp=timestamps[3]),  # Anomaly
                PerformanceMetric(name="throughput_items_per_second", value=102.0, timestamp=timestamps[4]),
                PerformanceMetric(name="throughput_items_per_second", value=103.0, timestamp=timestamps[5]),
                PerformanceMetric(name="throughput_items_per_second", value=101.0, timestamp=timestamps[6]),
                PerformanceMetric(name="throughput_items_per_second", value=99.0, timestamp=timestamps[7]),
                PerformanceMetric(name="throughput_items_per_second", value=100.0, timestamp=timestamps[8]),
                PerformanceMetric(name="throughput_items_per_second", value=102.0, timestamp=timestamps[9])
            ]
            
            # Add metrics to cache
            self.analyzer.metrics_cache["throughput_items_per_second"] = throughput_metrics
            
            # Run anomaly detection directly
            anomalies = self.analyzer._detect_anomalies(throughput_metrics)
            
            # Check that the anomaly was detected
            self.assertGreater(len(anomalies), 0)
            
            # Verify that the detected anomaly is the one we injected
            detected_anomaly = False
            for anomaly in anomalies:
                if abs(anomaly.value - 30.0) < 0.01:  # The anomalous value we injected
                    detected_anomaly = True
                    break
            
            self.assertTrue(detected_anomaly, "Failed to detect the injected anomaly")
            
            # Test alert generation
            alerts = self.analyzer._generate_alerts(anomalies, throughput_metrics)
            
            # Check that alerts were generated
            self.assertGreater(len(alerts), 0)
            
            # Verify the alert details
            found_alert = False
            for alert in alerts:
                if abs(alert.value - 30.0) < 0.01:
                    found_alert = True
                    # For throughput, lower is worse, so severity should be warning or critical
                    self.assertIn(alert.severity, ["warning", "critical"])
                    break
            
            self.assertTrue(found_alert, "Failed to generate an alert for the anomaly")
        finally:
            await self.async_tearDown()

    def test_analyzer_identifies_trends(self):
        anyio.run(self._test_analyzer_identifies_trends)

    async def _test_analyzer_identifies_trends(self):
        """Test that the analyzer successfully identifies trends in metrics."""
        # Set up
        await self.async_setUp()
        
        try:
            # Inject a series of metrics with a clear trend
            now = time.time()
            timestamps = [now + i * 10 for i in range(10)]  # 10 data points, 10 seconds apart
            
            # Create metrics for latency with an increasing trend (degradation)
            latency_metrics = [
                PerformanceMetric(name="latency_ms", value=100 + i * 5, timestamp=timestamps[i], model_name="bert-base")
                for i in range(10)
            ]
            
            # Add metrics to cache
            self.analyzer.metrics_cache["latency_ms"] = latency_metrics
            
            # Run trend analysis directly
            trend = self.analyzer._analyze_trend(latency_metrics)
            
            # Check that a trend was detected
            self.assertIsNotNone(trend)
            
            # Verify the trend details
            self.assertEqual(trend.metric_name, "latency_ms")
            self.assertEqual(trend.trend_type, "degrading")  # Increasing latency is degrading
            self.assertGreater(trend.confidence, 0.6)  # Should be high confidence
            self.assertEqual(trend.model_name, "bert-base")
            
            # Create metrics for throughput with an increasing trend (improvement)
            throughput_metrics = [
                PerformanceMetric(name="throughput_items_per_second", value=100 + i * 5, timestamp=timestamps[i], model_name="bert-base")
                for i in range(10)
            ]
            
            # Add metrics to cache
            self.analyzer.metrics_cache["throughput_items_per_second"] = throughput_metrics
            
            # Run trend analysis directly
            trend = self.analyzer._analyze_trend(throughput_metrics)
            
            # Check that a trend was detected
            self.assertIsNotNone(trend)
            
            # Verify the trend details
            self.assertEqual(trend.metric_name, "throughput_items_per_second")
            self.assertEqual(trend.trend_type, "improving")  # Increasing throughput is improving
            self.assertGreater(trend.confidence, 0.6)  # Should be high confidence
            self.assertEqual(trend.model_name, "bert-base")
        finally:
            await self.async_tearDown()

    def test_analyzer_generates_visualizations(self):
        anyio.run(self._test_analyzer_generates_visualizations)

    async def _test_analyzer_generates_visualizations(self):
        """Test that the analyzer successfully generates visualizations."""
        # Set up
        await self.async_setUp()
        
        try:
            # Inject metrics for visualization
            now = time.time()
            timestamps = [now - i * 10 for i in range(10)]  # 10 data points, 10 seconds apart
            
            # Create metrics for multiple models
            metrics = []
            
            # Latency metrics for bert-base
            for i in range(10):
                metrics.append(PerformanceMetric(
                    name="latency_ms",
                    value=100 + i * 2,  # Slight increase
                    timestamp=timestamps[i],
                    model_name="bert-base",
                    worker_id="worker1"
                ))
            
            # Latency metrics for vit-base
            for i in range(10):
                metrics.append(PerformanceMetric(
                    name="latency_ms",
                    value=120 - i * 1,  # Slight decrease
                    timestamp=timestamps[i],
                    model_name="vit-base",
                    worker_id="worker2"
                ))
            
            # Add metrics to cache
            self.analyzer._add_metrics_to_cache(metrics)
            
            # Clear any existing files in output directory
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            
            # Run visualization generation directly
            await self.analyzer._generate_visualizations()
            
            # Check that visualization files were created
            files = os.listdir(self.output_dir)
            self.assertGreater(len(files), 0)
            
            # Check for specific visualization files
            png_files = [f for f in files if f.endswith('.png')]
            self.assertGreater(len(png_files), 0)
            
            # Check for report file
            txt_files = [f for f in files if f.endswith('.txt')]
            self.assertGreater(len(txt_files), 0)
            
            # Check the content of the report file
            report_file = os.path.join(self.output_dir, txt_files[0])
            with open(report_file, 'r') as f:
                report_content = f.read()
            
            self.assertIn("PERFORMANCE TREND ANALYSIS REPORT", report_content)
            self.assertIn("PERFORMANCE TRENDS", report_content)
            
            # Verify that the report mentions the models
            self.assertIn("bert-base", report_content)
            self.assertIn("vit-base", report_content)
        finally:
            await self.async_tearDown()

    def test_database_integration(self):
        anyio.run(self._test_database_integration)

    async def _test_database_integration(self):
        """Test that the analyzer successfully stores and retrieves data from the database."""
        # Set up
        await self.async_setUp()
        
        try:
            # Check if DuckDB is available
            try:
                import duckdb
            except ImportError:
                self.skipTest("DuckDB not available")
            
            # Inject some test metrics
            now = time.time()
            test_metrics = [
                PerformanceMetric(
                    name="test_metric",
                    value=100.0,
                    timestamp=now,
                    task_id="task1",
                    worker_id="worker1",
                    model_name="test-model"
                ),
                PerformanceMetric(
                    name="test_metric",
                    value=105.0,
                    timestamp=now + 1,
                    task_id="task2",
                    worker_id="worker1",
                    model_name="test-model"
                )
            ]
            
            # Save metrics to database
            self.analyzer._save_metrics_to_db(test_metrics)
            
            # Save a test alert
            test_alert = PerformanceAlert(
                metric_name="test_metric",
                value=200.0,
                expected_range=(90.0, 110.0),
                severity="critical",
                timestamp=now,
                description="Test alert",
                alert_type="test",
                deviation_percent=100.0,
                task_id="task3",
                worker_id="worker1",
                model_name="test-model"
            )
            
            self.analyzer._save_alerts_to_db([test_alert])
            
            # Save a test trend
            test_trend = PerformanceTrend(
                metric_name="test_metric",
                trend_coefficient=1.0,
                trend_type="degrading",
                confidence=0.9,
                start_timestamp=now - 100,
                end_timestamp=now,
                data_points=10,
                description="Test trend",
                model_name="test-model"
            )
            
            self.analyzer._save_trend_to_db(test_trend)
            
            # Query the database to verify data was saved
            db = duckdb.connect(self.db_path)
            
            # Check metrics
            result = db.execute("SELECT COUNT(*) FROM performance_metrics WHERE name = 'test_metric'").fetchone()
            self.assertGreaterEqual(result[0], 2)
            
            # Check alerts
            result = db.execute("SELECT COUNT(*) FROM performance_alerts WHERE metric_name = 'test_metric'").fetchone()
            self.assertGreaterEqual(result[0], 1)
            
            # Check trends
            result = db.execute("SELECT COUNT(*) FROM performance_trends WHERE metric_name = 'test_metric'").fetchone()
            self.assertGreaterEqual(result[0], 1)
            
            # Verify metric values
            result = db.execute("SELECT value FROM performance_metrics WHERE name = 'test_metric' ORDER BY value").fetchall()
            values = [row[0] for row in result]
            self.assertIn(100.0, values)
            self.assertIn(105.0, values)
            
            # Verify alert severity
            result = db.execute("SELECT severity FROM performance_alerts WHERE metric_name = 'test_metric'").fetchone()
            self.assertEqual(result[0], "critical")
            
            # Verify trend type
            result = db.execute("SELECT trend_type FROM performance_trends WHERE metric_name = 'test_metric'").fetchone()
            self.assertEqual(result[0], "degrading")
            
            # Close the database connection
            db.close()
        finally:
            await self.async_tearDown()


def run_tests():
    """Run the integration tests."""
    try:
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_connects_to_coordinator'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_collects_metrics'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_detects_anomalies'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_identifies_trends'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_generates_visualizations'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_database_integration'))
        
        # Run tests
        runner = unittest.TextTestRunner()
        runner.run(suite)
    except Exception as e:
        print(f"Error running tests: {e}")


if __name__ == "__main__":
    run_tests()