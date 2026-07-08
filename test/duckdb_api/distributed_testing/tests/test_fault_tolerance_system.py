#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Fault Tolerance System component.
"""

import unittest
import os
import sys
import json
import time
import anyio
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from fault_tolerance_system import (
    FaultToleranceSystem,
    ErrorSeverity,
    ErrorCategory,
    RecoveryAction
)


class TestFaultToleranceSystem(unittest.TestCase):
    """Test suite for FaultToleranceSystem class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_coordinator = MagicMock()
        self.mock_task_manager = MagicMock()
        self.mock_worker_manager = MagicMock()
        
        # Initialize system with test settings
        self.fault_system = FaultToleranceSystem(
            coordinator=self.mock_coordinator,
            task_manager=self.mock_task_manager,
            worker_manager=self.mock_worker_manager,
            max_retries=3,
            circuit_break_threshold=2,
            circuit_break_timeout=1,  # 1 second for faster testing
            error_window_size=10,
            error_rate_threshold=0.5
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop fault tolerance system
        self.fault_system.stop()

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.fault_system.coordinator, self.mock_coordinator)
        self.assertEqual(self.fault_system.task_manager, self.mock_task_manager)
        self.assertEqual(self.fault_system.worker_manager, self.mock_worker_manager)
        
        self.assertEqual(self.fault_system.max_retries, 3)
        self.assertEqual(self.fault_system.circuit_break_threshold, 2)
        self.assertEqual(self.fault_system.circuit_break_timeout, 1)
        self.assertEqual(self.fault_system.error_window_size, 10)
        self.assertEqual(self.fault_system.error_rate_threshold, 0.5)
        
        self.assertIsNotNone(self.fault_system.worker_errors)
        self.assertIsNotNone(self.fault_system.task_errors)
        self.assertIsNotNone(self.fault_system.circuit_breakers)
        self.assertIsNotNone(self.fault_system.error_history)
        self.assertIsNotNone(self.fault_system.retry_counts)
        self.assertIsNotNone(self.fault_system.fallbacks)
        self.assertIsNotNone(self.fault_system.recovery_strategies)
        
        # Verify recovery strategies were initialized
        self.assertIn(ErrorCategory.NETWORK, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.RESOURCE, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.WORKER, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.TASK, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.DATA, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.HARDWARE, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.AUTHENTICATION, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.AUTHORIZATION, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.TIMEOUT, self.fault_system.recovery_strategies)
        self.assertIn(ErrorCategory.UNKNOWN, self.fault_system.recovery_strategies)

    def test_categorize_error(self):
        """Test error categorization."""
        # Network error
        error = ConnectionError("Failed to connect to worker")
        context = {}
        error_info = self.fault_system._categorize_error(error, context)
        
        self.assertEqual(error_info["category"], ErrorCategory.NETWORK)
        self.assertEqual(error_info["severity"], ErrorSeverity.MEDIUM)
        
        # Resource error
        error = MemoryError("Out of memory")
        context = {}
        error_info = self.fault_system._categorize_error(error, context)
        
        self.assertEqual(error_info["category"], ErrorCategory.RESOURCE)
        self.assertEqual(error_info["severity"], ErrorSeverity.HIGH)
        
        # Worker error
        error = Exception("Worker failed")
        context = {"worker_id": "worker-1"}
        error_info = self.fault_system._categorize_error(error, context)
        
        self.assertEqual(error_info["category"], ErrorCategory.WORKER)
        self.assertEqual(error_info["severity"], ErrorSeverity.MEDIUM)
        
        # Task error
        error = Exception("Task execution failed")
        context = {"task_id": "task-1"}
        error_info = self.fault_system._categorize_error(error, context)
        
        self.assertEqual(error_info["category"], ErrorCategory.TASK)
        self.assertEqual(error_info["severity"], ErrorSeverity.MEDIUM)
        
        # Timeout error
        error = TimeoutError("Operation timed out")
        context = {}
        error_info = self.fault_system._categorize_error(error, context)
        
        self.assertEqual(error_info["category"], ErrorCategory.TIMEOUT)
        self.assertEqual(error_info["severity"], ErrorSeverity.MEDIUM)
        
        # Severity from context
        error = Exception("Some error")
        context = {"critical": True}
        error_info = self.fault_system._categorize_error(error, context)
        
        self.assertEqual(error_info["severity"], ErrorSeverity.CRITICAL)

    def test_handle_error_retry(self):
        """Test handling error with retry action."""
        # Network error should retry
        error = ConnectionError("Failed to connect to worker")
        context = {}
        
        action = self.fault_system.handle_error(error, context)
        
        self.assertEqual(action["action"], RecoveryAction.RETRY)
        self.assertIn("error_category", action)
        self.assertIn("error_severity", action)
        self.assertIn("retry_count", action)
        self.assertIn("operation_id", action)
        
        # Test retry with same operation ID
        operation_id = action["operation_id"]
        action = self.fault_system.handle_error(error, context, operation_id=operation_id)
        
        self.assertEqual(action["action"], RecoveryAction.RETRY)
        self.assertEqual(action["retry_count"], 1)  # Incremented
        
        # Third retry
        action = self.fault_system.handle_error(error, context, operation_id=operation_id)
        
        self.assertEqual(action["action"], RecoveryAction.RETRY)
        self.assertEqual(action["retry_count"], 2)  # Incremented again
        
        # Fourth try (exceeds max_retries)
        action = self.fault_system.handle_error(error, context, operation_id=operation_id)
        
        self.assertEqual(action["action"], RecoveryAction.CIRCUIT_BREAK)
        self.assertEqual(action["retry_count"], 3)  # Incremented to max

    def test_handle_error_circuit_break(self):
        """Test handling error with circuit breaking."""
        # Setup service
        service_key = "test_service"
        error = Exception("Service error")
        context = {"service_key": service_key}
        
        # First error
        action = self.fault_system.handle_error(error, context)
        
        self.assertEqual(action["action"], RecoveryAction.RETRY)
        self.assertFalse(self.fault_system._is_circuit_open(service_key))
        
        # Second error (reaches threshold)
        action = self.fault_system.handle_error(error, context)
        
        self.assertEqual(action["action"], RecoveryAction.RETRY)
        
        # Check circuit breaker directly
        with self.fault_system.circuit_breakers_lock:
            self.assertIn(service_key, self.fault_system.circuit_breakers)
            circuit = self.fault_system.circuit_breakers[service_key]
            self.assertEqual(circuit["error_count"], 2)
            
            # Manually open circuit for testing
            circuit["state"] = "open"
            
        # Error after circuit open
        action = self.fault_system.handle_error(error, context)
        
        self.assertEqual(action["action"], RecoveryAction.FALLBACK)
        self.assertEqual(action["reason"], "Circuit breaker open")

    def test_register_fallback(self):
        """Test registering a fallback."""
        service_key = "test_service"
        fallback = {"method": "mock_fallback"}
        
        self.fault_system.register_fallback(service_key, fallback)
        
        self.assertIn(service_key, self.fault_system.fallbacks)
        self.assertEqual(self.fault_system.fallbacks[service_key], fallback)
        
        # Test getting fallback
        result_fallback = self.fault_system._get_fallback(service_key)
        self.assertEqual(result_fallback, fallback)
        
        # Test getting non-existent fallback
        result_fallback = self.fault_system._get_fallback("non_existent")
        self.assertIsNone(result_fallback)

    def test_register_recovery_strategy(self):
        """Test registering a custom recovery strategy."""
        # Define a test strategy
        def test_strategy(severity, retry_count, context):
            return {
                "action": RecoveryAction.NOTIFY,
                "reason": "Test strategy"
            }
        
        # Register strategy
        self.fault_system.register_recovery_strategy(ErrorCategory.NETWORK, test_strategy)
        
        # Test strategy is used
        error = ConnectionError("Network error")
        context = {}
        
        action = self.fault_system.handle_error(error, context)
        
        self.assertEqual(action["action"], RecoveryAction.NOTIFY)
        self.assertEqual(action["reason"], "Test strategy")
        
        # Test registering with string
        self.fault_system.register_recovery_strategy("timeout", test_strategy)
        
        error = TimeoutError("Operation timed out")
        action = self.fault_system.handle_error(error, context)
        
        self.assertEqual(action["action"], RecoveryAction.NOTIFY)
        self.assertEqual(action["reason"], "Test strategy")

    def test_retry_operation(self):
        """Test retrying an operation."""
        # Mock function that fails twice then succeeds
        mock_func = MagicMock()
        mock_func.side_effect = [
            ConnectionError("First failure"),
            ConnectionError("Second failure"),
            "success"
        ]
        
        # Test retry
        success, result = self.fault_system.retry_operation(
            operation_func=mock_func,
            args=("arg1", "arg2"),
            kwargs={"key": "value"},
            context={},
            max_retries=3
        )
        
        self.assertTrue(success)
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 3)
        
        # Verify function was called with correct args
        mock_func.assert_called_with("arg1", "arg2", key="value")
        
        # Test with function that always fails
        mock_func.reset_mock()
        mock_func.side_effect = ConnectionError("Always fails")
        
        # Add fallback
        mock_fallback = MagicMock(return_value="fallback_result")
        
        success, result = self.fault_system.retry_operation(
            operation_func=mock_func,
            context={},
            max_retries=2,
            fallback_func=mock_fallback
        )
        
        self.assertTrue(success)
        self.assertEqual(result, "fallback_result")
        self.assertEqual(mock_func.call_count, 3)  # 3 attempts (original + 2 retries)
        self.assertEqual(mock_fallback.call_count, 1)

    def test_reset_circuit_breaker(self):
        """Test resetting a circuit breaker."""
        # Setup circuit breaker
        service_key = "test_service"
        with self.fault_system.circuit_breakers_lock:
            self.fault_system.circuit_breakers[service_key] = {
                "state": "open",
                "error_count": 5,
                "last_error_time": datetime.now(),
                "reset_time": None
            }
        
        # Reset circuit breaker
        result = self.fault_system.reset_circuit_breaker(service_key)
        
        self.assertTrue(result)
        
        # Verify circuit is reset
        with self.fault_system.circuit_breakers_lock:
            self.assertIn(service_key, self.fault_system.circuit_breakers)
            circuit = self.fault_system.circuit_breakers[service_key]
            self.assertEqual(circuit["state"], "closed")
            self.assertEqual(circuit["error_count"], 0)
            self.assertIsNone(circuit["last_error_time"])
            self.assertIsNotNone(circuit["reset_time"])
        
        # Test resetting non-existent circuit
        result = self.fault_system.reset_circuit_breaker("non_existent")
        self.assertFalse(result)

    def test_reset_retry_count(self):
        """Test resetting retry count."""
        # Setup retry count
        operation_id = "test_operation"
        with self.fault_system.retry_counts_lock:
            self.fault_system.retry_counts[operation_id] = 3
        
        # Reset retry count
        result = self.fault_system.reset_retry_count(operation_id)
        
        self.assertTrue(result)
        
        # Verify retry count is reset
        with self.fault_system.retry_counts_lock:
            self.assertIn(operation_id, self.fault_system.retry_counts)
            self.assertEqual(self.fault_system.retry_counts[operation_id], 0)
        
        # Test resetting non-existent retry count
        result = self.fault_system.reset_retry_count("non_existent")
        self.assertFalse(result)

    def test_get_error_statistics(self):
        """Test getting error statistics."""
        # Add some errors
        worker_id = "worker-1"
        task_id = "task-1"
        
        # Worker error
        error_info = {
            "error": Exception("Worker error"),
            "error_type": "Exception",
            "error_message": "Worker error",
            "category": ErrorCategory.WORKER,
            "severity": ErrorSeverity.MEDIUM,
            "timestamp": datetime.now(),
            "context": {"worker_id": worker_id}
        }
        self.fault_system._record_error(error_info, worker_id=worker_id)
        
        # Task error
        error_info = {
            "error": Exception("Task error"),
            "error_type": "Exception",
            "error_message": "Task error",
            "category": ErrorCategory.TASK,
            "severity": ErrorSeverity.HIGH,
            "timestamp": datetime.now() - timedelta(hours=1),
            "context": {"task_id": task_id}
        }
        self.fault_system._record_error(error_info, task_id=task_id)
        
        # Network error
        error_info = {
            "error": ConnectionError("Network error"),
            "error_type": "ConnectionError",
            "error_message": "Network error",
            "category": ErrorCategory.NETWORK,
            "severity": ErrorSeverity.MEDIUM,
            "timestamp": datetime.now(),
            "context": {}
        }
        self.fault_system._record_error(error_info)
        
        # Get worker statistics
        stats = self.fault_system.get_error_statistics(worker_id=worker_id)
        
        self.assertEqual(stats["total_errors"], 1)
        self.assertEqual(stats["categories"][ErrorCategory.WORKER.value], 1)
        self.assertEqual(stats["severities"][ErrorSeverity.MEDIUM.value], 1)
        
        # Get task statistics
        stats = self.fault_system.get_error_statistics(task_id=task_id)
        
        self.assertEqual(stats["total_errors"], 1)
        self.assertEqual(stats["categories"][ErrorCategory.TASK.value], 1)
        self.assertEqual(stats["severities"][ErrorSeverity.HIGH.value], 1)
        
        # Get global statistics
        stats = self.fault_system.get_error_statistics()
        
        self.assertEqual(stats["total_errors"], 3)
        self.assertEqual(stats["categories"][ErrorCategory.WORKER.value], 1)
        self.assertEqual(stats["categories"][ErrorCategory.TASK.value], 1)
        self.assertEqual(stats["categories"][ErrorCategory.NETWORK.value], 1)
        self.assertEqual(stats["error_rate"], 3/10)  # 3 errors, window size 10
        self.assertIn("time_distribution", stats)

    def test_circuit_breaker_timeout(self):
        """Test circuit breaker automatic timeout."""
        # Setup circuit breaker
        service_key = "test_service"
        with self.fault_system.circuit_breakers_lock:
            self.fault_system.circuit_breakers[service_key] = {
                "state": "open",
                "error_count": 5,
                "last_error_time": datetime.now() - timedelta(seconds=2),  # 2 seconds ago (longer than timeout)
                "reset_time": None
            }
        
        # Wait for monitor thread to reset
        time.sleep(1.5)  # A bit longer than circuit_break_timeout
        
        # Verify circuit is reset to half-open
        with self.fault_system.circuit_breakers_lock:
            self.assertIn(service_key, self.fault_system.circuit_breakers)
            circuit = self.fault_system.circuit_breakers[service_key]
            self.assertEqual(circuit["state"], "half-open")
            self.assertIsNotNone(circuit["reset_time"])


if __name__ == '__main__':
    unittest.main()