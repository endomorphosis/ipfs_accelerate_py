#!/usr/bin/env python3
"""
Test module for distributed error handler functionality.
"""

import unittest
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.duckdb.distributed_testing.distributed_error_handler import (
    DistributedErrorHandler,
    ErrorCategory,
    RetryPolicy,
    ResourceRecoveryStrategy,
    NetworkRecoveryStrategy,
    HardwareRecoveryStrategy,
    WorkerRecoveryStrategy,
    TestExecutionRecoveryStrategy,
    ErrorAggregator
)

# Disable logging for tests
logging.disable(logging.CRITICAL)

class TestErrorCategory(unittest.TestCase):
    """Test the ErrorCategory enumeration."""
    
    def test_category_values(self):
        """Test that all categories have expected values."""
        self.assertEqual(ErrorCategory.RESOURCE_EXHAUSTED.value, "resource_exhausted")
        self.assertEqual(ErrorCategory.NETWORK_TIMEOUT.value, "network_timeout")
        self.assertEqual(ErrorCategory.TEST_ASSERTION_ERROR.value, "test_assertion_error")
        self.assertEqual(ErrorCategory.HARDWARE_NOT_AVAILABLE.value, "hardware_not_available")
        self.assertEqual(ErrorCategory.WORKER_DISCONNECTED.value, "worker_disconnected")
        self.assertEqual(ErrorCategory.UNKNOWN.value, "unknown")


class TestRetryPolicy(unittest.TestCase):
    """Test the RetryPolicy class."""
    
    def test_should_retry_basic(self):
        """Test basic retry logic."""
        policy = RetryPolicy(max_retries=3)
        
        # Under max retries
        self.assertTrue(policy.should_retry(ErrorCategory.UNKNOWN, 1))
        self.assertTrue(policy.should_retry(ErrorCategory.UNKNOWN, 2))
        self.assertTrue(policy.should_retry(ErrorCategory.UNKNOWN, 3))
        
        # Exceeds max retries
        self.assertFalse(policy.should_retry(ErrorCategory.UNKNOWN, 4))
    
    def test_should_retry_with_categories(self):
        """Test retry with specific categories."""
        policy = RetryPolicy(
            max_retries=3,
            categories_to_retry=[ErrorCategory.NETWORK_TIMEOUT],
            categories_to_skip=[ErrorCategory.TEST_ASSERTION_ERROR]
        )
        
        # Category to retry
        self.assertTrue(policy.should_retry(ErrorCategory.NETWORK_TIMEOUT, 1))
        
        # Category to skip
        self.assertFalse(policy.should_retry(ErrorCategory.TEST_ASSERTION_ERROR, 1))
        
        # Other category (not in either list)
        self.assertFalse(policy.should_retry(ErrorCategory.UNKNOWN, 1))
    
    def test_get_retry_delay(self):
        """Test retry delay calculation."""
        policy = RetryPolicy(
            retry_delay_seconds=10,
            retry_backoff_factor=2,
            retry_jitter=0
        )
        
        # First attempt
        self.assertEqual(policy.get_retry_delay(1), 10)
        
        # Second attempt (backoff factor applied)
        self.assertEqual(policy.get_retry_delay(2), 20)
        
        # Third attempt (backoff factor applied again)
        self.assertEqual(policy.get_retry_delay(3), 40)
    
    def test_max_retry_delay(self):
        """Test maximum retry delay limit."""
        policy = RetryPolicy(
            retry_delay_seconds=10,
            retry_backoff_factor=10,
            max_retry_delay_seconds=100,
            retry_jitter=0
        )
        
        # First attempt
        self.assertEqual(policy.get_retry_delay(1), 10)
        
        # Second attempt (backoff factor applied)
        self.assertEqual(policy.get_retry_delay(2), 100)  # Capped at max_retry_delay_seconds
        
        # Third attempt (still capped)
        self.assertEqual(policy.get_retry_delay(3), 100)


class TestRecoveryStrategies(unittest.TestCase):
    """Test recovery strategy implementations."""
    
    def test_resource_recovery_strategy(self):
        """Test resource recovery strategy."""
        strategy = ResourceRecoveryStrategy()
        
        # Test applicability
        self.assertTrue(strategy.is_applicable(ErrorCategory.RESOURCE_EXHAUSTED))
        self.assertTrue(strategy.is_applicable(ErrorCategory.RESOURCE_UNAVAILABLE))
        self.assertTrue(strategy.is_applicable(ErrorCategory.RESOURCE_NOT_FOUND))
        self.assertFalse(strategy.is_applicable(ErrorCategory.NETWORK_TIMEOUT))
        
        # Test recovery
        error = {"category": ErrorCategory.RESOURCE_EXHAUSTED}
        context = {"worker_id": "worker1"}
        
        result = strategy.recover("task1", error, context)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy"], "resource")
        self.assertIn("request_resource_cleanup", result["actions_taken"])
        self.assertTrue(result["retry_recommended"])
    
    def test_network_recovery_strategy(self):
        """Test network recovery strategy."""
        strategy = NetworkRecoveryStrategy()
        
        # Test applicability
        self.assertTrue(strategy.is_applicable(ErrorCategory.NETWORK_TIMEOUT))
        self.assertTrue(strategy.is_applicable(ErrorCategory.NETWORK_CONNECTION_ERROR))
        self.assertTrue(strategy.is_applicable(ErrorCategory.NETWORK_SERVER_ERROR))
        self.assertFalse(strategy.is_applicable(ErrorCategory.RESOURCE_EXHAUSTED))
        
        # Test recovery for timeout
        error = {"category": ErrorCategory.NETWORK_TIMEOUT}
        context = {"worker_id": "worker1"}
        
        result = strategy.recover("task1", error, context)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy"], "network")
        self.assertIn("increase_timeout", result["actions_taken"])
        self.assertTrue(result["retry_recommended"])
        
        # Test recovery for connection error
        error = {"category": ErrorCategory.NETWORK_CONNECTION_ERROR}
        result = strategy.recover("task1", error, context)
        
        self.assertTrue(result["success"])
        self.assertIn("reconnect", result["actions_taken"])
        self.assertTrue(result["retry_recommended"])
    
    def test_hardware_recovery_strategy(self):
        """Test hardware recovery strategy."""
        strategy = HardwareRecoveryStrategy()
        
        # Test applicability
        self.assertTrue(strategy.is_applicable(ErrorCategory.HARDWARE_NOT_AVAILABLE))
        self.assertTrue(strategy.is_applicable(ErrorCategory.HARDWARE_MISMATCH))
        self.assertTrue(strategy.is_applicable(ErrorCategory.HARDWARE_COMPATIBILITY_ERROR))
        self.assertFalse(strategy.is_applicable(ErrorCategory.RESOURCE_EXHAUSTED))
        
        # Test recovery
        error = {"category": ErrorCategory.HARDWARE_NOT_AVAILABLE}
        context = {"worker_id": "worker1", "hardware_type": "cuda"}
        
        result = strategy.recover("task1", error, context)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy"], "hardware")
        self.assertIn("mark_hardware_unavailable:cuda:worker1", result["actions_taken"])
        self.assertIn("reallocate_to_alternative_hardware", result["actions_taken"])
        self.assertTrue(result["retry_recommended"])
    
    def test_worker_recovery_strategy(self):
        """Test worker recovery strategy."""
        strategy = WorkerRecoveryStrategy()
        
        # Test applicability
        self.assertTrue(strategy.is_applicable(ErrorCategory.WORKER_DISCONNECTED))
        self.assertTrue(strategy.is_applicable(ErrorCategory.WORKER_TIMEOUT))
        self.assertTrue(strategy.is_applicable(ErrorCategory.WORKER_CRASHED))
        self.assertFalse(strategy.is_applicable(ErrorCategory.RESOURCE_EXHAUSTED))
        
        # Test recovery
        error = {"category": ErrorCategory.WORKER_DISCONNECTED}
        context = {"worker_id": "worker1"}
        
        result = strategy.recover("task1", error, context)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy"], "worker")
        self.assertIn("mark_worker_unavailable:worker1", result["actions_taken"])
        self.assertIn("reassign_task", result["actions_taken"])
        self.assertTrue(result["retry_recommended"])
    
    def test_test_execution_recovery_strategy(self):
        """Test test execution recovery strategy."""
        strategy = TestExecutionRecoveryStrategy()
        
        # Test applicability
        self.assertTrue(strategy.is_applicable(ErrorCategory.TEST_ASSERTION_ERROR))
        self.assertTrue(strategy.is_applicable(ErrorCategory.TEST_IMPORT_ERROR))
        self.assertTrue(strategy.is_applicable(ErrorCategory.TEST_DEPENDENCY_ERROR))
        self.assertTrue(strategy.is_applicable(ErrorCategory.TEST_SYNTAX_ERROR))
        self.assertFalse(strategy.is_applicable(ErrorCategory.RESOURCE_EXHAUSTED))
        
        # Test recovery for import error
        error = {"category": ErrorCategory.TEST_IMPORT_ERROR}
        context = {}
        
        result = strategy.recover("task1", error, context)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy"], "test_execution")
        self.assertIn("check_dependencies", result["actions_taken"])
        self.assertTrue(result["retry_recommended"])
        
        # Test recovery for assertion error (should not retry)
        error = {"category": ErrorCategory.TEST_ASSERTION_ERROR}
        result = strategy.recover("task1", error, context)
        
        self.assertTrue(result["success"])
        self.assertIn("record_test_failure", result["actions_taken"])
        self.assertFalse(result["retry_recommended"])


class TestErrorAggregator(unittest.TestCase):
    """Test the ErrorAggregator class."""
    
    def test_add_error_new_group(self):
        """Test adding error to a new group."""
        aggregator = ErrorAggregator()
        
        error = {
            "type": "ConnectionError",
            "message": "Connection refused",
            "category": ErrorCategory.NETWORK_CONNECTION_ERROR
        }
        
        group_id = aggregator.add_error(error)
        
        self.assertEqual(group_id, 1)
        self.assertEqual(len(aggregator.error_groups), 1)
        self.assertEqual(len(aggregator.error_groups[1]), 1)
        self.assertEqual(aggregator.error_groups[1][0], error)
    
    def test_add_error_existing_group(self):
        """Test adding similar error to existing group."""
        aggregator = ErrorAggregator()
        
        error1 = {
            "type": "ConnectionError",
            "message": "Connection refused by server",
            "category": ErrorCategory.NETWORK_CONNECTION_ERROR
        }
        
        error2 = {
            "type": "ConnectionError",
            "message": "Connection refused by remote host",
            "category": ErrorCategory.NETWORK_CONNECTION_ERROR
        }
        
        group_id1 = aggregator.add_error(error1)
        group_id2 = aggregator.add_error(error2)
        
        self.assertEqual(group_id1, group_id2)
        self.assertEqual(len(aggregator.error_groups), 1)
        self.assertEqual(len(aggregator.error_groups[group_id1]), 2)
    
    def test_add_error_different_groups(self):
        """Test adding different errors to different groups."""
        aggregator = ErrorAggregator()
        
        error1 = {
            "type": "ConnectionError",
            "message": "Connection refused",
            "category": ErrorCategory.NETWORK_CONNECTION_ERROR
        }
        
        error2 = {
            "type": "MemoryError",
            "message": "Out of memory",
            "category": ErrorCategory.RESOURCE_EXHAUSTED
        }
        
        group_id1 = aggregator.add_error(error1)
        group_id2 = aggregator.add_error(error2)
        
        self.assertNotEqual(group_id1, group_id2)
        self.assertEqual(len(aggregator.error_groups), 2)
        
    def test_get_frequent_errors(self):
        """Test getting frequent errors."""
        aggregator = ErrorAggregator()
        
        # Add multiple similar errors
        base_error = {
            "type": "ConnectionError",
            "message": "Connection refused",
            "category": ErrorCategory.NETWORK_CONNECTION_ERROR
        }
        
        for i in range(5):
            aggregator.add_error(base_error.copy())
        
        # Add a different error
        other_error = {
            "type": "MemoryError",
            "message": "Out of memory",
            "category": ErrorCategory.RESOURCE_EXHAUSTED
        }
        
        aggregator.add_error(other_error)
        
        # Get frequent errors with min_count=3
        frequent_errors = aggregator.get_frequent_errors(min_count=3)
        
        self.assertEqual(len(frequent_errors), 1)
        group_id, count, error = frequent_errors[0]
        self.assertEqual(count, 5)
        self.assertEqual(error["type"], "ConnectionError")


class TestDistributedErrorHandler(unittest.TestCase):
    """Test the DistributedErrorHandler class."""
    
    def setUp(self):
        """Set up test case."""
        self.handler = DistributedErrorHandler()
    
    def test_categorize_error_by_type(self):
        """Test error categorization by type."""
        error = {"type": "MemoryError", "message": "Out of memory"}
        
        category = self.handler.categorize_error(error)
        
        self.assertEqual(category, ErrorCategory.RESOURCE_EXHAUSTED)
    
    def test_categorize_error_by_message(self):
        """Test error categorization by message pattern."""
        # Type not in known types
        error = {"type": "CustomError", "message": "Connection timed out while waiting for response"}
        
        category = self.handler.categorize_error(error)
        
        self.assertEqual(category, ErrorCategory.NETWORK_TIMEOUT)
    
    def test_categorize_error_unknown(self):
        """Test error categorization for unknown error."""
        error = {"type": "CustomError", "message": "Something unexpected happened"}
        
        category = self.handler.categorize_error(error)
        
        self.assertEqual(category, ErrorCategory.UNKNOWN)
    
    def test_should_retry(self):
        """Test retry decision making."""
        error_retry = {
            "type": "ConnectionError",
            "message": "Connection refused",
            "category": ErrorCategory.NETWORK_CONNECTION_ERROR
        }
        
        error_no_retry = {
            "type": "AssertionError",
            "message": "Test assertion failed",
            "category": ErrorCategory.TEST_ASSERTION_ERROR
        }
        
        # Test retryable error
        should_retry, delay = self.handler.should_retry("task1", error_retry, 1)
        self.assertTrue(should_retry)
        self.assertIsNotNone(delay)
        
        # Test non-retryable error
        should_retry, delay = self.handler.should_retry("task1", error_no_retry, 1)
        self.assertFalse(should_retry)
        self.assertIsNone(delay)
        
        # Test max retries exceeded
        should_retry, delay = self.handler.should_retry("task1", error_retry, 10)
        self.assertFalse(should_retry)
        self.assertIsNone(delay)
    
    def test_get_recovery_strategy(self):
        """Test getting recovery strategy for error category."""
        # Resource error
        strategy = self.handler.get_recovery_strategy(ErrorCategory.RESOURCE_EXHAUSTED)
        self.assertIsInstance(strategy, ResourceRecoveryStrategy)
        
        # Network error
        strategy = self.handler.get_recovery_strategy(ErrorCategory.NETWORK_TIMEOUT)
        self.assertIsInstance(strategy, NetworkRecoveryStrategy)
        
        # Hardware error
        strategy = self.handler.get_recovery_strategy(ErrorCategory.HARDWARE_NOT_AVAILABLE)
        self.assertIsInstance(strategy, HardwareRecoveryStrategy)
        
        # Worker error
        strategy = self.handler.get_recovery_strategy(ErrorCategory.WORKER_DISCONNECTED)
        self.assertIsInstance(strategy, WorkerRecoveryStrategy)
        
        # Test error
        strategy = self.handler.get_recovery_strategy(ErrorCategory.TEST_ASSERTION_ERROR)
        self.assertIsInstance(strategy, TestExecutionRecoveryStrategy)
    
    def test_handle_error_complete(self):
        """Test complete error handling flow."""
        error = {
            "type": "ConnectionError",
            "message": "Connection refused by remote host",
            "traceback": "...stack trace..."
        }
        
        context = {
            "worker_id": "worker1",
            "hardware_type": "cuda",
            "attempt_count": 1
        }
        
        result = self.handler.handle_error("task1", error, context)
        
        # Verify error was categorized
        self.assertEqual(result["error_category"], ErrorCategory.NETWORK_CONNECTION_ERROR)
        
        # Verify retry decision
        self.assertTrue(result["retry"])
        self.assertIsNotNone(result["retry_delay"])
        
        # Verify recovery action
        self.assertIsNotNone(result["recovery_action"])
        self.assertEqual(result["recovery_action"]["strategy"], "network")
        self.assertIn("reconnect", result["recovery_action"]["actions_taken"])
        
        # Verify success indicator
        self.assertTrue(result["error_handled"])


if __name__ == "__main__":
    unittest.main()