#!/usr/bin/env python3
"""
Test script for the Automated Optimization Manager.

This script tests the automated performance optimization features, including:
- Performance issue detection
- Automatic optimization actions
- Trend analysis
- Scheduled optimization tasks
"""

import os
import sys
import time
import json
import logging
import unittest
import tempfile
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import modules to test
from duckdb_api.simulation_validation.automated_optimization_manager import (
    AutomatedOptimizationManager,
    get_optimization_manager
)
from duckdb_api.simulation_validation.db_performance_optimizer import DBPerformanceOptimizer


class TestAutomatedOptimizationManager(unittest.TestCase):
    """Test cases for the AutomatedOptimizationManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock DBPerformanceOptimizer
        self.mock_optimizer = MagicMock(spec=DBPerformanceOptimizer)
        
        # Configure the mock to return sample metrics
        self.sample_metrics = {
            "query_time": {
                "value": 300.0,
                "unit": "ms",
                "status": "good"
            },
            "storage_size": {
                "value": 419430400,  # 400 MB
                "unit": "bytes",
                "status": "good"
            },
            "index_efficiency": {
                "value": 85.0,
                "unit": "percent",
                "status": "good"
            },
            "vacuum_status": {
                "value": 75.0,
                "unit": "percent",
                "status": "good"
            },
            "compression_ratio": {
                "value": 3.5,
                "unit": "ratio",
                "status": "good"
            },
            "read_efficiency": {
                "value": 230.0,
                "unit": "records/second",
                "status": "good"
            },
            "write_efficiency": {
                "value": 180.0,
                "unit": "records/second",
                "status": "good"
            },
            "cache_performance": {
                "value": 65.0,
                "unit": "percent",
                "status": "good"
            }
        }
        
        self.mock_optimizer.get_performance_metrics.return_value = self.sample_metrics
        self.mock_optimizer.get_overall_status.return_value = "good"
        
        # Configure cleanup_old_records to return sample data
        self.mock_optimizer.cleanup_old_records.return_value = {
            "table1": {"count": 100, "deleted": 100},
            "table2": {"count": 200, "deleted": 200}
        }
        
        # Set up other methods to return success
        self.mock_optimizer.create_indexes.return_value = True
        self.mock_optimizer.analyze_tables.return_value = True
        self.mock_optimizer.optimize_database.return_value = True
        self.mock_optimizer.clear_cache.return_value = None
        
        # Create the optimization manager with the mock
        self.manager = AutomatedOptimizationManager(
            db_optimizer=self.mock_optimizer,
            check_interval=1,  # 1 second for faster testing
            auto_apply_actions=False,
            enable_scheduled_tasks=False
        )
        
        # Create a temp directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = os.path.join(self.temp_dir.name, "optimization.log")
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop monitoring if active
        if self.manager.monitoring_active:
            self.manager.stop_monitoring()
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of the manager."""
        # Test with default thresholds
        self.assertEqual(self.manager.thresholds, AutomatedOptimizationManager.DEFAULT_THRESHOLDS)
        self.assertEqual(self.manager.actions, AutomatedOptimizationManager.DEFAULT_ACTIONS)
        self.assertFalse(self.manager.auto_apply_actions)
        
        # Test with custom thresholds and actions
        custom_thresholds = {
            "query_time": {"warning": 400.0, "error": 800.0}
        }
        custom_actions = {
            "query_time": ["create_indexes"]
        }
        
        manager = AutomatedOptimizationManager(
            db_optimizer=self.mock_optimizer,
            thresholds=custom_thresholds,
            actions=custom_actions
        )
        
        self.assertEqual(manager.thresholds, custom_thresholds)
        self.assertEqual(manager.actions, custom_actions)
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        # Create manager with log file
        manager_with_log = AutomatedOptimizationManager(
            db_optimizer=self.mock_optimizer,
            log_file=self.log_file
        )
        
        # Check if the log file attribute is set
        self.assertEqual(manager_with_log.log_file, self.log_file)
        
        # Log a test message
        logger = logging.getLogger("automated_optimization_manager")
        logger.info("Test log message")
        
        # Check if the message was written to the log file
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Test log message", log_content)
    
    def test_detect_performance_issues_no_issues(self):
        """Test detecting performance issues when all metrics are good."""
        issues = self.manager._detect_performance_issues(self.sample_metrics)
        self.assertEqual(len(issues), 0, "No issues should be detected with good metrics")
    
    def test_detect_performance_issues_with_issues(self):
        """Test detecting performance issues when metrics exceed thresholds."""
        # Create metrics with issues
        metrics_with_issues = self.sample_metrics.copy()
        
        # Modify metrics to exceed thresholds
        metrics_with_issues["query_time"]["value"] = 600.0  # Exceeds warning threshold (500)
        metrics_with_issues["storage_size"]["value"] = 1073741824 + 1  # Exceeds error threshold (1 GB)
        metrics_with_issues["index_efficiency"]["value"] = 60.0  # Exceeds warning threshold (70)
        
        # Detect issues
        issues = self.manager._detect_performance_issues(metrics_with_issues)
        
        # Verify issues were detected
        self.assertEqual(len(issues), 3, "Should detect 3 issues")
        self.assertIn("query_time", issues, "Should detect query_time issue")
        self.assertIn("storage_size", issues, "Should detect storage_size issue")
        self.assertIn("index_efficiency", issues, "Should detect index_efficiency issue")
        
        # Verify severity levels
        self.assertEqual(issues["query_time"]["severity"], "warning")
        self.assertEqual(issues["storage_size"]["severity"], "error")
        self.assertEqual(issues["index_efficiency"]["severity"], "warning")
        
        # Verify recommended actions
        self.assertEqual(issues["query_time"]["recommended_actions"], 
                        self.manager.actions["query_time"])
        self.assertEqual(issues["storage_size"]["recommended_actions"], 
                        self.manager.actions["storage_size"])
    
    def test_execute_actions(self):
        """Test executing optimization actions."""
        # Define actions to test
        actions = ["create_indexes", "analyze_tables", "cleanup_old_records", "optimize_database", "clear_cache"]
        
        # Execute actions
        results = self.manager.execute_actions(actions)
        
        # Verify all actions were executed
        for action in actions:
            self.assertIn(action, results, f"Action {action} should be in results")
            self.assertEqual(results[action]["status"], "success", f"Action {action} should succeed")
        
        # Verify method calls
        self.mock_optimizer.create_indexes.assert_called_once()
        self.mock_optimizer.analyze_tables.assert_called_once()
        self.mock_optimizer.cleanup_old_records.assert_called_once()
        self.mock_optimizer.optimize_database.assert_called_once()
        self.mock_optimizer.clear_cache.assert_called_once()
    
    def test_execute_actions_with_error(self):
        """Test executing actions when an error occurs."""
        # Configure the mock to raise an exception
        self.mock_optimizer.create_indexes.side_effect = Exception("Test error")
        
        # Execute actions
        results = self.manager.execute_actions(["create_indexes", "analyze_tables"])
        
        # Verify error was captured
        self.assertEqual(results["create_indexes"]["status"], "error")
        self.assertIn("Test error", results["create_indexes"]["message"])
        
        # Verify other action was still executed
        self.assertEqual(results["analyze_tables"]["status"], "success")
    
    def test_check_performance(self):
        """Test checking database performance."""
        # Check performance
        result = self.manager.check_performance()
        
        # Verify result structure
        self.assertIn("metrics", result)
        self.assertIn("issues", result)
        self.assertIn("overall_status", result)
        self.assertIn("timestamp", result)
        
        # Verify metrics are included
        self.assertEqual(result["metrics"], self.sample_metrics)
        
        # Verify overall status
        self.assertEqual(result["overall_status"], "good")
        
        # Check with specific metrics
        result = self.manager.check_performance(["query_time", "storage_size"])
        self.assertEqual(len(result["metrics"]), 2)
        self.assertIn("query_time", result["metrics"])
        self.assertIn("storage_size", result["metrics"])
    
    def test_optimize_now(self):
        """Test running optimization immediately."""
        # Create metrics with issues
        metrics_with_issues = self.sample_metrics.copy()
        metrics_with_issues["query_time"]["value"] = 600.0  # Exceeds warning threshold
        self.mock_optimizer.get_performance_metrics.return_value = metrics_with_issues
        
        # Configure get_overall_status to return different values for before and after
        self.mock_optimizer.get_overall_status.side_effect = ["warning", "good"]
        
        # Run optimization
        result = self.manager.optimize_now()
        
        # Verify result structure
        self.assertIn("status", result)
        self.assertIn("results", result)
        self.assertIn("before_overall_status", result)
        self.assertIn("after_overall_status", result)
        self.assertIn("timestamp", result)
        
        # Verify optimization results
        self.assertEqual(result["status"], "success")
        self.assertIn("query_time", result["results"])
        self.assertEqual(result["before_overall_status"], "warning")
        self.assertEqual(result["after_overall_status"], "good")
    
    def test_run_comprehensive_optimization(self):
        """Test running comprehensive optimization."""
        # Configure metrics before and after optimization
        before_metrics = self.sample_metrics.copy()
        before_metrics["query_time"]["value"] = 400.0
        
        after_metrics = before_metrics.copy()
        after_metrics["query_time"]["value"] = 200.0
        
        # Configure the mock to return different metrics before and after
        self.mock_optimizer.get_performance_metrics.side_effect = [before_metrics, after_metrics]
        
        # Run comprehensive optimization
        result = self.manager.run_comprehensive_optimization()
        
        # Verify result structure
        self.assertIn("status", result)
        self.assertIn("action_results", result)
        self.assertIn("improvements", result)
        
        # Verify optimization results
        self.assertEqual(result["status"], "success")
        self.assertIn("query_time", result["improvements"])
        self.assertEqual(result["improvements"]["query_time"]["before"], 400.0)
        self.assertEqual(result["improvements"]["query_time"]["after"], 200.0)
        self.assertEqual(result["improvements"]["query_time"]["improvement"], 50.0)  # 50% improvement
    
    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        # Create metrics with issues
        metrics_with_issues = self.sample_metrics.copy()
        metrics_with_issues["storage_size"]["value"] = 600000000  # Exceeds warning threshold
        self.mock_optimizer.get_performance_metrics.return_value = metrics_with_issues
        
        # Get recommendations
        result = self.manager.get_optimization_recommendations()
        
        # Verify result structure
        self.assertIn("status", result)
        self.assertIn("recommendations", result)
        self.assertIn("overall_status", result)
        
        # Verify recommendations
        self.assertEqual(result["status"], "success")
        self.assertIn("storage_size", result["recommendations"])
        self.assertEqual(result["recommendations"]["storage_size"]["recommended_actions"], 
                      self.manager.actions["storage_size"])
    
    def test_scheduled_tasks(self):
        """Test scheduling and managing tasks."""
        # Create manager with scheduled tasks
        manager = AutomatedOptimizationManager(
            db_optimizer=self.mock_optimizer,
            enable_scheduled_tasks=True
        )
        
        # Check if default tasks were created
        tasks = manager.get_scheduled_tasks()
        self.assertIn("daily_optimization", tasks)
        self.assertIn("weekly_optimization", tasks)
        self.assertIn("monthly_optimization", tasks)
        
        # Schedule a custom task
        task = manager.schedule_task(
            task_name="custom_task",
            actions=["optimize_database"],
            interval_hours=12
        )
        
        # Verify task was scheduled
        self.assertIn("custom_task", manager.get_scheduled_tasks())
        
        # Disable a task
        manager.enable_task("custom_task", False)
        self.assertFalse(manager.scheduled_tasks["custom_task"]["enabled"])
        
        # Enable a task
        manager.enable_task("custom_task", True)
        self.assertTrue(manager.scheduled_tasks["custom_task"]["enabled"])
        
        # Unschedule a task
        manager.unschedule_task("custom_task")
        self.assertNotIn("custom_task", manager.get_scheduled_tasks())
    
    def test_check_scheduled_tasks(self):
        """Test checking scheduled tasks for execution."""
        # Create a task that should run immediately
        past_time = datetime.now() - timedelta(hours=1)
        task = self.manager.schedule_task(
            task_name="immediate_task",
            actions=["create_indexes"],
            interval_hours=24,
            start_time=past_time
        )
        
        # Create a task that should not run yet
        future_time = datetime.now() + timedelta(hours=1)
        task = self.manager.schedule_task(
            task_name="future_task",
            actions=["analyze_tables"],
            interval_hours=24,
            start_time=future_time
        )
        
        # Check scheduled tasks
        self.manager._check_scheduled_tasks()
        
        # Verify the immediate task was executed
        self.assertIsNotNone(self.manager.scheduled_tasks["immediate_task"]["last_run"])
        self.assertGreater(self.manager.scheduled_tasks["immediate_task"]["next_run"], datetime.now())
        
        # Verify the future task was not executed
        self.assertIsNone(self.manager.scheduled_tasks["future_task"]["last_run"])
        self.assertEqual(self.manager.scheduled_tasks["future_task"]["next_run"], future_time)
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.manager.start_monitoring()
        self.assertTrue(self.manager.monitoring_active)
        self.assertIsNotNone(self.manager.monitoring_thread)
        self.assertTrue(self.manager.monitoring_thread.is_alive())
        
        # Stop monitoring
        self.manager.stop_monitoring()
        self.assertFalse(self.manager.monitoring_active)
        
        # Ensure thread terminates within reasonable time
        self.manager.monitoring_thread.join(timeout=5.0)
        self.assertFalse(self.manager.monitoring_thread.is_alive())
    
    def test_monitoring_loop(self):
        """Test the monitoring loop functionality."""
        # Configure the mock to return metrics with issues
        metrics_with_issues = self.sample_metrics.copy()
        metrics_with_issues["query_time"]["value"] = 600.0  # Exceeds warning threshold
        self.mock_optimizer.get_performance_metrics.return_value = metrics_with_issues
        
        # Enable auto-apply
        self.manager.auto_apply_actions = True
        
        # Run the monitoring loop manually for a short time
        self.manager.monitoring_active = True
        monitoring_thread = threading.Thread(target=self.manager._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        # Let the loop run for a short time
        time.sleep(2)
        
        # Stop the loop
        self.manager.monitoring_active = False
        monitoring_thread.join(timeout=2.0)
        
        # Verify that actions were executed
        self.mock_optimizer.create_indexes.assert_called()
        self.mock_optimizer.analyze_tables.assert_called()
    
    def test_update_metrics_history(self):
        """Test updating metrics history."""
        # Update metrics history
        self.manager._update_metrics_history(self.sample_metrics)
        
        # Verify history was updated
        history = self.manager.get_metrics_history()
        for metric_name in self.sample_metrics:
            self.assertIn(metric_name, history)
            self.assertEqual(len(history[metric_name]), 1)
            self.assertEqual(history[metric_name][0]["value"], self.sample_metrics[metric_name]["value"])
        
        # Update again and verify history grows
        self.manager._update_metrics_history(self.sample_metrics)
        history = self.manager.get_metrics_history()
        for metric_name in self.sample_metrics:
            self.assertEqual(len(history[metric_name]), 2)
    
    def test_analyze_trends(self):
        """Test analyzing performance trends."""
        # Create some historical data
        now = datetime.now()
        
        # Seven days of data (one per day)
        for days_ago in range(7, 0, -1):
            timestamp = (now - timedelta(days=days_ago)).isoformat()
            
            # Create metrics for this day
            day_metrics = self.sample_metrics.copy()
            
            # Make query_time increasing (worse)
            day_metrics["query_time"]["value"] = 200.0 + days_ago * 30
            
            # Make index_efficiency decreasing (worse)
            day_metrics["index_efficiency"]["value"] = 90.0 - days_ago * 2
            
            # Update history
            for metric_name, metric_data in day_metrics.items():
                if metric_name not in self.manager.metrics_history:
                    self.manager.metrics_history[metric_name] = []
                
                self.manager.metrics_history[metric_name].append({
                    "timestamp": timestamp,
                    "value": metric_data["value"],
                    "status": metric_data["status"]
                })
        
        # Analyze trends
        result = self.manager.analyze_trends(days=7)
        
        # Verify result structure
        self.assertIn("status", result)
        self.assertIn("trends", result)
        self.assertIn("period_days", result)
        
        # Verify trend analysis
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["period_days"], 7)
        
        # Check query_time trend (should be increasing)
        self.assertIn("query_time", result["trends"])
        self.assertEqual(result["trends"]["query_time"]["direction"], "decreasing")
        self.assertTrue(result["trends"]["query_time"]["change_percent"] > 0)
        
        # Check index_efficiency trend (should be increasing)
        self.assertIn("index_efficiency", result["trends"])
        self.assertEqual(result["trends"]["index_efficiency"]["direction"], "increasing")
        self.assertTrue(result["trends"]["index_efficiency"]["change_percent"] > 0)
        
        # Check if trends are concerning
        self.assertFalse(result["trends"]["query_time"]["concerning"])
        self.assertFalse(result["trends"]["index_efficiency"]["concerning"])
    
    def test_get_optimization_manager_utility(self):
        """Test the utility function for creating a manager."""
        # Create a temporary config file
        config_file = os.path.join(self.temp_dir.name, "config.json")
        config = {
            "thresholds": {
                "query_time": {"warning": 400.0, "error": 800.0}
            },
            "check_interval": 300,
            "auto_apply_actions": True
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Create manager with config file
        manager = get_optimization_manager(
            db_optimizer=self.mock_optimizer,
            config_file=config_file
        )
        
        # Verify configuration was loaded
        self.assertEqual(manager.thresholds["query_time"]["warning"], 400.0)
        self.assertEqual(manager.check_interval, 300)
        self.assertTrue(manager.auto_apply_actions)
        
        # Test with non-existent config file
        manager = get_optimization_manager(
            db_optimizer=self.mock_optimizer,
            config_file="nonexistent.json",
            auto_apply=True
        )
        
        # Verify defaults were used but auto_apply was set
        self.assertEqual(manager.thresholds, AutomatedOptimizationManager.DEFAULT_THRESHOLDS)
        self.assertTrue(manager.auto_apply_actions)


def main():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    main()