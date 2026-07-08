#!/usr/bin/env python3
"""
Unit tests for the Performance-Based Error Recovery System

This module contains tests to verify the functionality of the performance-based
error recovery system for the Distributed Testing Framework.

Run with: python -m unittest distributed_testing/tests/test_error_recovery_performance.py
"""

import unittest
import anyio
import tempfile
import os
import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import duckdb
import sys
import pathlib

# Ensure the module directory is in the path for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from ..error_recovery_with_performance_tracking import (
    PerformanceBasedErrorRecovery,
    RecoveryPerformanceRecord,
    RecoveryPerformanceMetric,
    ProgressiveRecoveryLevel
)

from ..distributed_error_handler import (
    DistributedErrorHandler,
    ErrorReport,
    ErrorContext,
    ErrorType,
    ErrorSeverity
)

from ..error_recovery_strategies import (
    EnhancedErrorRecoveryManager,
    RecoveryStrategy,
    ErrorCategory
)


class MockRecoveryStrategy(RecoveryStrategy):
    """Mock recovery strategy for testing."""
    
    def __init__(self, name="test", level="low", success=True, delay=0.1):
        self.name = name
        self.level = level
        self.success = success
        self.delay = delay
        self.executed = False
        self.args = None
    
    async def execute(self, error_info):
        """Mock execution."""
        self.executed = True
        self.args = error_info
        await anyio.sleep(self.delay)
        return self.success


class MockDistributedErrorHandler(DistributedErrorHandler):
    """Mock error handler for testing."""
    
    def __init__(self):
        super().__init__()
        self.errors = {}
        self.registered_hooks = {}
    
    def register_error_hook(self, error_type, hook):
        """Register an error hook."""
        self.registered_hooks[error_type] = hook
    
    def create_error_report(self, exception, context_data=None):
        """Create a mock error report."""
        context_data = context_data or {}
        
        error_id = f"test_error_{len(self.errors)}"
        
        context = ErrorContext(
            component=context_data.get("component", "test"),
            operation=context_data.get("operation", "test"),
            user_id=context_data.get("user_id"),
            request_id=context_data.get("request_id"),
            environment=context_data.get("environment", {}),
            stack_trace=None,
            related_entities=context_data.get("related_entities", {}),
            metadata=context_data.get("metadata", {})
        )
        
        report = ErrorReport(
            error_id=error_id,
            error_type=context_data.get("error_type", ErrorType.UNKNOWN),
            error_severity=context_data.get("error_severity", ErrorSeverity.MEDIUM),
            message=str(exception),
            context=context,
            exception=exception
        )
        
        self.errors[error_id] = report
        return report


class MockCoordinator:
    """Mock coordinator for testing."""
    
    def __init__(self, db_connection=None):
        """Initialize the mock coordinator."""
        self.tasks = {}
        self.running_tasks = {}
        self.pending_tasks = set()
        self.worker_connections = {}
        self.db = db_connection
        
        # Add some test tasks
        for i in range(10):
            task_id = f"task-{i}"
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": "pending" if i < 3 else "running" if i < 8 else "completed"
            }
            
            if self.tasks[task_id]["status"] == "running":
                self.running_tasks[task_id] = f"worker-{i % 3}"
            elif self.tasks[task_id]["status"] == "pending":
                self.pending_tasks.add(task_id)


class MockEnhancedErrorRecoveryManager(EnhancedErrorRecoveryManager):
    """Mock recovery manager for testing."""
    
    def __init__(self, coordinator=None):
        """Initialize the mock recovery manager."""
        self.coordinator = coordinator
        self.strategies = {}
        
        # Add some test strategies
        self.strategies["retry"] = MockRecoveryStrategy("retry", "low", True, 0.1)
        self.strategies["worker_restart"] = MockRecoveryStrategy("worker_restart", "medium", True, 0.2)
        self.strategies["db_reconnect"] = MockRecoveryStrategy("db_reconnect", "medium", True, 0.3)
        self.strategies["coordinator_restart"] = MockRecoveryStrategy("coordinator_restart", "high", True, 0.4)
        self.strategies["system_recovery"] = MockRecoveryStrategy("system_recovery", "critical", True, 0.5)


class TestPerformanceBasedErrorRecovery(unittest.TestCase):
    """Tests for the PerformanceBasedErrorRecovery class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database path.
        # DuckDB expects to initialize the file itself; connecting to an existing
        # empty file can raise "not a valid DuckDB database file".
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.duckdb")

        # Connect to database
        self.db_connection = duckdb.connect(self.db_path)
        
        # Create mock components
        self.coordinator = MockCoordinator(self.db_connection)
        self.error_handler = MockDistributedErrorHandler()
        self.recovery_manager = MockEnhancedErrorRecoveryManager(self.coordinator)
        
        # Create performance-based recovery system
        self.recovery_system = PerformanceBasedErrorRecovery(
            error_handler=self.error_handler,
            recovery_manager=self.recovery_manager,
            coordinator=self.coordinator,
            db_connection=self.db_connection
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Close database connection
        self.db_connection.close()

        # Cleanup temporary directory (and database file)
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of the recovery system."""
        # Check that database tables were created
        tables = self.db_connection.execute("""
        SELECT name FROM sqlite_master WHERE type='table'
        """).fetchall()
        
        table_names = [t[0] for t in tables]
        
        self.assertIn('recovery_performance', table_names)
        self.assertIn('strategy_scores', table_names)
        self.assertIn('adaptive_timeouts', table_names)
        self.assertIn('progressive_recovery', table_names)
    
    def test_recover_success(self):
        anyio.run(self._test_recover_success)

    async def _test_recover_success(self):
        """Test successful recovery."""
        # Create error report
        error_report = self.error_handler.create_error_report(
            Exception("Test error"),
            {"error_type": ErrorType.NETWORK, "component": "test"}
        )
        
        # Reset strategy execution flags
        for strategy in self.recovery_manager.strategies.values():
            strategy.executed = False
        
        # Mock strategy to ensure success
        strategy = self.recovery_manager.strategies["retry"]
        strategy.success = True
        
        # Test recovery
        success, info = await self.recovery_system.recover(error_report)
        
        # Check results
        self.assertTrue(success)
        self.assertEqual(info["strategy_id"], "retry")
        self.assertEqual(info["error_type"], "network")
        self.assertTrue(strategy.executed)
        
        # Check that performance was recorded
        records = self.db_connection.execute("""
        SELECT * FROM recovery_performance
        """).fetchall()
        
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1], "retry")  # strategy_id
        self.assertEqual(records[0][2], "retry")  # strategy_name
        self.assertEqual(records[0][3], "network")  # error_type
        self.assertEqual(records[0][5], True)  # success
    
    def test_recover_failure(self):
        anyio.run(self._test_recover_failure)

    async def _test_recover_failure(self):
        """Test failed recovery."""
        # Create error report
        error_report = self.error_handler.create_error_report(
            Exception("Test error"),
            {"error_type": ErrorType.NETWORK, "component": "test"}
        )
        
        # Force all strategies to fail so progressive recovery escalates to max
        for strategy in self.recovery_manager.strategies.values():
            strategy.executed = False
            strategy.success = False
        
        # Test recovery
        success, info = await self.recovery_system.recover(error_report)
        
        # Check results (implementation escalates within a single call)
        self.assertFalse(success)
        self.assertEqual(info["error_type"], "network")
        self.assertEqual(info["recovery_level"], ProgressiveRecoveryLevel.LEVEL_5.value)
        self.assertEqual(info["strategy_id"], "system_recovery")
        
        # Check that performance was recorded
        records = self.db_connection.execute("""
        SELECT * FROM recovery_performance
        """).fetchall()
        
        # One record per level (LEVEL_1..LEVEL_5)
        self.assertEqual(len(records), ProgressiveRecoveryLevel.LEVEL_5.value)
        self.assertEqual(records[-1][1], "system_recovery")  # strategy_id
        self.assertEqual(records[-1][2], "system_recovery")  # strategy_name
        self.assertEqual(records[-1][3], "network")  # error_type
        self.assertEqual(records[-1][5], False)  # success
    
    def test_progressive_recovery(self):
        anyio.run(self._test_progressive_recovery)

    async def _test_progressive_recovery(self):
        """Test progressive recovery with escalation."""
        # Create error report
        error_report = self.error_handler.create_error_report(
            Exception("Test error"),
            {"error_type": ErrorType.NETWORK, "component": "test"}
        )
        
        # Force all strategies to fail
        for strategy in self.recovery_manager.strategies.values():
            strategy.executed = False
            strategy.success = False

        # Test recovery (should escalate through all levels within one call)
        success, info = await self.recovery_system.recover(error_report)

        self.assertFalse(success)
        self.assertEqual(info["recovery_level"], ProgressiveRecoveryLevel.LEVEL_5.value)
        self.assertEqual(info["strategy_id"], "system_recovery")

        # Max level should be recorded for this error
        self.assertEqual(
            self.recovery_system.error_recovery_levels[error_report.error_id],
            ProgressiveRecoveryLevel.LEVEL_5.value,
        )
        
        # Check that progression was recorded
        records = self.db_connection.execute("""
        SELECT error_id, current_level, history FROM progressive_recovery
        """).fetchall()
        
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], error_report.error_id)  # error_id
        self.assertEqual(records[0][1], ProgressiveRecoveryLevel.LEVEL_5.value)  # current_level
        
        # Check history JSON
        history = json.loads(records[0][2])
        self.assertEqual(len(history), ProgressiveRecoveryLevel.LEVEL_5.value - 1)
        self.assertEqual(history[0]["old_level"], 1)
        self.assertEqual(history[0]["new_level"], 2)
        self.assertEqual(history[-1]["old_level"], 4)
        self.assertEqual(history[-1]["new_level"], 5)
    
    def test_strategy_selection(self):
        anyio.run(self._test_strategy_selection)

    async def _test_strategy_selection(self):
        """Test strategy selection based on performance history."""
        # Record fake performance history
        self.recovery_system._record_performance(
            strategy_id="retry",
            strategy_name="retry",
            error_type="network",
            execution_time=0.1,
            success=True,
            affected_tasks=5,
            recovered_tasks=5
        )
        
        self.recovery_system._record_performance(
            strategy_id="worker_restart",
            strategy_name="worker_restart",
            error_type="network",
            execution_time=0.2,
            success=False,
            affected_tasks=5,
            recovered_tasks=0
        )
        
        # Update strategy scores
        self.recovery_system._update_strategy_scores("retry", "network")
        self.recovery_system._update_strategy_scores("worker_restart", "network")
        
        # Create error report
        error_report = self.error_handler.create_error_report(
            Exception("Test error"),
            {"error_type": ErrorType.NETWORK, "component": "test"}
        )
        
        # Test strategy selection
        strategy, strategy_id = await self.recovery_system._select_best_strategy("network", 1)
        
        # Check that the better performing strategy was selected
        self.assertEqual(strategy_id, "retry")
    
    def test_adaptive_timeouts(self):
        anyio.run(self._test_adaptive_timeouts)

    async def _test_adaptive_timeouts(self):
        """Test adaptive timeouts."""
        # Set initial timeout
        key = "network:retry"
        self.recovery_system.adaptive_timeouts[key] = 10.0
        
        # Record successful execution (should decrease timeout)
        self.recovery_system._update_adaptive_timeout("network", "retry", 10.0, True)
        
        # Check that timeout decreased
        self.assertLess(self.recovery_system.adaptive_timeouts[key], 10.0)
        
        # Record timeout failure (should increase timeout)
        original_timeout = self.recovery_system.adaptive_timeouts[key]
        self.recovery_system._update_adaptive_timeout("network", "retry", original_timeout, False)
        
        # Check that timeout increased
        self.assertGreater(self.recovery_system.adaptive_timeouts[key], original_timeout)
        
        # Check database persistence
        record = self.db_connection.execute("""
        SELECT timeout FROM adaptive_timeouts
        WHERE error_type = 'network' AND strategy_id = 'retry'
        """).fetchone()
        
        self.assertIsNotNone(record)
        self.assertEqual(record[0], self.recovery_system.adaptive_timeouts[key])
    
    def test_resource_monitoring(self):
        anyio.run(self._test_resource_monitoring)

    async def _test_resource_monitoring(self):
        """Test resource usage monitoring."""
        # Get initial resource usage
        resources = self.recovery_system._get_resource_usage()
        
        # Check that required metrics are present
        self.assertIn("cpu_percent", resources)
        self.assertIn("memory_percent", resources)
        self.assertIn("process_memory_mb", resources)
        
        # Test resource diff calculation
        before = {"cpu_percent": 10.0, "memory_percent": 20.0}
        after = {"cpu_percent": 15.0, "memory_percent": 25.0}
        
        diff = self.recovery_system._calculate_resource_diff(before, after)
        
        self.assertEqual(diff["cpu_percent"], 5.0)
        self.assertEqual(diff["memory_percent"], 5.0)
    
    def test_impact_score(self):
        anyio.run(self._test_impact_score)

    async def _test_impact_score(self):
        """Test impact score calculation."""
        # Test with various inputs
        score1 = self.recovery_system._calculate_impact_score(
            resource_diff={"cpu_percent": 5.0, "memory_percent": 10.0},
            execution_time=1.0,
            affected_tasks=2,
            recovered_tasks=2
        )
        
        # Higher resource usage and longer time
        score2 = self.recovery_system._calculate_impact_score(
            resource_diff={"cpu_percent": 20.0, "memory_percent": 30.0},
            execution_time=10.0,
            affected_tasks=2,
            recovered_tasks=2
        )
        
        # Failed recovery
        score3 = self.recovery_system._calculate_impact_score(
            resource_diff={"cpu_percent": 5.0, "memory_percent": 10.0},
            execution_time=1.0,
            affected_tasks=5,
            recovered_tasks=0
        )
        
        # Check that higher resource usage and time increases score
        self.assertLess(score1, score2)
        
        # Check that failed recovery increases score
        self.assertLess(score1, score3)
    
    def test_stability_check(self):
        anyio.run(self._test_stability_check)

    async def _test_stability_check(self):
        """Test system stability check."""
        # Test stability check
        stability = await self.recovery_system._check_stability()
        
        # Should return a value between 0 and 1
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
    
    def test_performance_metrics(self):
        anyio.run(self._test_performance_metrics)

    async def _test_performance_metrics(self):
        """Test performance metrics collection."""
        # Record some performance data
        self.recovery_system._record_performance(
            strategy_id="retry",
            strategy_name="retry",
            error_type="network",
            execution_time=0.1,
            success=True,
            affected_tasks=5,
            recovered_tasks=5
        )
        
        self.recovery_system._record_performance(
            strategy_id="retry",
            strategy_name="retry",
            error_type="timeout",
            execution_time=0.2,
            success=False,
            affected_tasks=5,
            recovered_tasks=0
        )
        
        self.recovery_system._record_performance(
            strategy_id="worker_restart",
            strategy_name="worker_restart",
            error_type="network",
            execution_time=0.3,
            success=True,
            affected_tasks=5,
            recovered_tasks=4
        )
        
        # Update strategy scores
        self.recovery_system._update_strategy_scores("retry", "network")
        self.recovery_system._update_strategy_scores("retry", "timeout")
        self.recovery_system._update_strategy_scores("worker_restart", "network")
        
        # Get metrics
        metrics = self.recovery_system.get_performance_metrics()
        
        # Check metrics
        self.assertEqual(metrics["overall"]["total_executions"], 3)
        self.assertEqual(metrics["overall"]["strategy_count"], 2)
        self.assertEqual(metrics["overall"]["error_type_count"], 2)
        
        # Check strategies
        self.assertEqual(len(metrics["strategies"]), 2)
        self.assertIn("retry", metrics["strategies"])
        self.assertIn("worker_restart", metrics["strategies"])
        
        # Check top strategies
        self.assertEqual(len(metrics["top_strategies"]), 2)
        self.assertIn("network", metrics["top_strategies"])
        self.assertIn("timeout", metrics["top_strategies"])
    
    def test_error_recovery_with_affected_tasks(self):
        anyio.run(self._test_error_recovery_with_affected_tasks)

    async def _test_error_recovery_with_affected_tasks(self):
        """Test recovery with affected tasks detection."""
        # Create error report with task info
        error_report = self.error_handler.create_error_report(
            Exception("Test error"),
            {
                "error_type": ErrorType.WORKER,
                "component": "worker",
                "operation": "execute",
                "related_entities": {
                    "task_id": "task-1",
                    "worker_id": "worker-1"
                }
            }
        )
        
        # Get affected tasks
        affected_tasks = await self.recovery_system._get_affected_tasks(error_report)
        
        # Should include the specific task
        self.assertIn("task-1", affected_tasks)
        
        # Create error report for worker error (affects all tasks for that worker)
        error_report2 = self.error_handler.create_error_report(
            Exception("Worker offline"),
            {
                "error_type": ErrorType.WORKER,
                "component": "worker",
                "operation": "heartbeat",
                "related_entities": {
                    "worker_id": "worker-1"
                }
            }
        )
        
        # Reset running tasks
        self.coordinator.running_tasks = {
            "task-1": "worker-1",
            "task-2": "worker-1",
            "task-3": "worker-2"
        }
        
        # Get affected tasks
        affected_tasks2 = await self.recovery_system._get_affected_tasks(error_report2)
        
        # Should include all tasks for worker-1
        self.assertIn("task-1", affected_tasks2)
        self.assertIn("task-2", affected_tasks2)
        self.assertNotIn("task-3", affected_tasks2)  # Different worker
    
    def test_task_recovery_tracking(self):
        anyio.run(self._test_task_recovery_tracking)

    async def _test_task_recovery_tracking(self):
        """Test tracking of task recovery."""
        # Set up some affected tasks
        affected_tasks = {"task-1", "task-2", "task-3"}
        
        # Modify task statuses
        self.coordinator.tasks["task-1"]["status"] = "running"  # Recovered
        self.coordinator.tasks["task-2"]["status"] = "failed"   # Not recovered
        self.coordinator.tasks["task-3"]["status"] = "pending"  # Recovered
        
        # Check task recovery
        recovered = await self.recovery_system._check_task_recovery(affected_tasks)
        
        # Should indicate 2 recovered tasks
        self.assertEqual(recovered, 2)
        
        # Test retry scenario
        self.coordinator.tasks["task-2"]["status"] = "failed"
        self.coordinator.tasks["task-2"]["retried_task_id"] = "task-4"
        self.coordinator.tasks["task-4"] = {"status": "running"}
        
        # Check task recovery again
        recovered = await self.recovery_system._check_task_recovery(affected_tasks)
        
        # Now all 3 should be recovered (task-2 was retried as task-4)
        self.assertEqual(recovered, 3)
    
    def test_strategy_recommendations(self):
        """Test strategy recommendations."""
        # Record performance history
        self.recovery_system._record_performance(
            strategy_id="retry",
            strategy_name="retry",
            error_type="network",
            execution_time=0.1,
            success=True,
            affected_tasks=5,
            recovered_tasks=5
        )
        
        self.recovery_system._record_performance(
            strategy_id="worker_restart",
            strategy_name="worker_restart",
            error_type="network",
            execution_time=0.3,
            success=True,
            affected_tasks=5,
            recovered_tasks=4
        )
        
        # Update strategy scores
        self.recovery_system._update_strategy_scores("retry", "network")
        self.recovery_system._update_strategy_scores("worker_restart", "network")
        
        # Get recommendations
        recommendations = self.recovery_system.get_strategy_recommendations("network")
        
        # Should have recommendations
        self.assertEqual(len(recommendations), 2)
        
        # Should be sorted by score (highest first)
        self.assertEqual(recommendations[0]["strategy_id"], "retry")
    
    def test_reset_recovery_level(self):
        """Test resetting recovery level."""
        # Set a recovery level
        error_id = "test_error"
        self.recovery_system.error_recovery_levels[error_id] = 3
        
        # Reset recovery level
        result = self.recovery_system.reset_recovery_level(error_id)
        
        # Should succeed
        self.assertTrue(result)
        
        # Level should be removed
        self.assertNotIn(error_id, self.recovery_system.error_recovery_levels)
        
        # Check database removal
        record = self.db_connection.execute("""
        SELECT * FROM progressive_recovery WHERE error_id = ?
        """, (error_id,)).fetchone()
        
        self.assertIsNone(record)
    
    def test_reset_all_recovery_levels(self):
        """Test resetting all recovery levels."""
        # Set some recovery levels
        self.recovery_system.error_recovery_levels["error1"] = 2
        self.recovery_system.error_recovery_levels["error2"] = 3
        
        # Reset all levels
        result = self.recovery_system.reset_all_recovery_levels()
        
        # Should succeed
        self.assertTrue(result)
        
        # All levels should be removed
        self.assertEqual(len(self.recovery_system.error_recovery_levels), 0)
        
        # Check database removal
        count = self.db_connection.execute("""
        SELECT COUNT(*) FROM progressive_recovery
        """).fetchone()[0]
        
        self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()