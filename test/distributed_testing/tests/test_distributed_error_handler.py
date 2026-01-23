#!/usr/bin/env python3
"""
Tests for the Distributed Error Handler

This module contains unit tests for the distributed error handler implementation.
"""

import pytest
import anyio
import time
from unittest.mock import Mock, patch
import logging
from datetime import datetime, timedelta

from ..distributed_error_handler import (
    DistributedErrorHandler, ErrorType, ErrorSeverity, 
    ErrorContext, ErrorReport, RetryPolicy, ErrorAggregator,
    safe_execute, safe_execute_async
)


@pytest.fixture
def error_handler():
    """Create a fresh error handler for testing."""
    handler = DistributedErrorHandler()
    # Mock the database connection to avoid actual DB operations
    handler.db_connection = Mock()
    handler._persist_error = Mock()
    return handler


@pytest.fixture
def mock_exception():
    """Create a standard exception for testing."""
    return ValueError("Test error message")


@pytest.fixture
def error_context():
    """Create a standard error context for testing."""
    return {
        "component": "test_component",
        "operation": "test_operation",
        "user_id": "test_user",
        "request_id": "test_request",
        "environment": {"test_env": "value"},
        "metadata": {"test_meta": "value"}
    }


def test_error_categorization(error_handler, mock_exception, error_context):
    """Test that errors are correctly categorized by type and severity."""
    error_type, error_severity = error_handler.categorize_error(mock_exception, error_context)
    
    assert isinstance(error_type, ErrorType)
    assert isinstance(error_severity, ErrorSeverity)
    assert error_type == ErrorType.VALIDATION  # ValueError should be categorized as VALIDATION
    
    # Test different error types
    network_error = ConnectionError("Failed to connect")
    error_type, _ = error_handler.categorize_error(network_error, error_context)
    assert error_type == ErrorType.NETWORK
    
    timeout_error = TimeoutError("Operation timed out")
    error_type, _ = error_handler.categorize_error(timeout_error, error_context)
    assert error_type == ErrorType.TIMEOUT
    
    # Test severity adjustment based on context
    high_priority_context = {**error_context, "high_priority": True}
    _, error_severity = error_handler.categorize_error(mock_exception, high_priority_context)
    assert error_severity == ErrorSeverity.HIGH


def test_error_report_creation(error_handler, mock_exception, error_context):
    """Test creating detailed error reports."""
    error_report = error_handler.create_error_report(mock_exception, error_context)
    
    # Verify error report fields
    assert isinstance(error_report, ErrorReport)
    assert error_report.error_id.startswith("err_")
    assert error_report.message == str(mock_exception)
    assert error_report.error_type == ErrorType.VALIDATION
    assert error_report.context.component == error_context["component"]
    assert error_report.context.operation == error_context["operation"]
    assert error_report.context.user_id == error_context["user_id"]
    assert error_report.context.stack_trace is not None
    
    # Verify the error was stored
    assert error_report.error_id in error_handler.errors


def test_error_aggregation(error_handler):
    """Test that similar errors are properly aggregated."""
    # Create several similar errors
    error1 = error_handler.create_error_report(
        ValueError("Common error"), 
        {"component": "component1", "operation": "operation1"}
    )
    
    error2 = error_handler.create_error_report(
        ValueError("Common error"), 
        {"component": "component1", "operation": "operation1"}
    )
    
    error3 = error_handler.create_error_report(
        ValueError("Different message"), 
        {"component": "component1", "operation": "operation1"}
    )
    
    # Get group for the first error
    group = error_handler.error_aggregator.get_group(error1.error_id)
    assert group is not None
    
    # Check if the second error was aggregated with the first one
    assert any(e.error_id == error2.error_id for e in group)
    
    # Check the primary error's aggregation count
    primary = error_handler.error_aggregator.get_primary_error(error1.error_id)
    assert primary.aggregated_count > 1


def test_retry_policy_calculation():
    """Test retry policy delay calculations."""
    # Create a retry policy with jitter disabled for consistent testing
    policy = RetryPolicy(
        max_retries=3,
        initial_delay_ms=100,
        backoff_factor=2.0,
        jitter=False,  # Disable jitter for predictable values
        skip_on_error_types=[ErrorType.AUTHENTICATION],
        skip_if_severity_above=ErrorSeverity.HIGH
    )
    
    # Test retry timeout calculation
    delay = policy.get_delay_ms(0)
    assert delay == 100  # Should be exactly 100 with jitter disabled
    
    delay = policy.get_delay_ms(1)
    assert delay == 200  # Should be exactly 200 (100*2^1) with jitter disabled
    
    delay = policy.get_delay_ms(2)
    assert delay == 400  # Should be exactly 400 (100*2^2) with jitter disabled
    
    # Test policy attributes
    assert policy.max_retries == 3
    assert policy.initial_delay_ms == 100
    assert policy.backoff_factor == 2.0
    assert policy.jitter is False
    assert ErrorType.AUTHENTICATION in policy.skip_on_error_types
    assert policy.skip_if_severity_above == ErrorSeverity.HIGH
    
    # Create a policy with jitter enabled
    policy_with_jitter = RetryPolicy(
        max_retries=3,
        initial_delay_ms=100,
        backoff_factor=2.0,
        jitter=True,
        skip_on_error_types=[ErrorType.AUTHENTICATION]
    )
    
    # Test that jitter adds some variation (should be in 75-125ms range)
    delay = policy_with_jitter.get_delay_ms(0)
    assert 75 <= delay <= 125
    
    # Test to_dict method
    policy_dict = policy.to_dict()
    assert policy_dict["max_retries"] == 3
    assert policy_dict["initial_delay_ms"] == 100
    assert "authentication" in policy_dict["skip_on_error_types"]
    assert policy_dict["skip_if_severity_above"] == "high"


@pytest.mark.asyncio
async def test_retry_operation_success(error_handler):
    """Test successful retry after initial failures."""
    # Mock function that fails twice then succeeds
    attempt = 0
    def flaky_function():
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            raise ConnectionError(f"Failed attempt {attempt}")
        return "Success"
    
    # Set up retry policy
    error_handler.set_retry_policy("test_component", RetryPolicy(
        max_retries=3,
        initial_delay_ms=50,  # Short delay for tests
        backoff_factor=1.0
    ))
    
    # Execute with retry
    result, error_report = await error_handler.retry_operation(
        flaky_function,
        context={"component": "test_component", "operation": "flaky_operation"}
    )
    
    # Check result
    assert result == "Success"
    assert error_report is None
    assert attempt == 3  # Function should have been called 3 times


@pytest.mark.asyncio
async def test_retry_operation_failure(error_handler):
    """Test retry operation that ultimately fails."""
    # Mock function that always fails
    def failing_function():
        raise ValueError("Always fails")
    
    # Set up retry policy
    error_handler.set_retry_policy("test_component", RetryPolicy(
        max_retries=2,
        initial_delay_ms=50,  # Short delay for tests
        backoff_factor=1.0
    ))
    
    # Execute with retry
    result, error_report = await error_handler.retry_operation(
        failing_function,
        context={"component": "test_component", "operation": "failing_operation"}
    )
    
    # Check result
    assert result is None
    assert error_report is not None
    assert error_report.error_type == ErrorType.VALIDATION
    assert error_report.retry_count == 2
    assert error_report.resolution_status == "failed"


def test_retry_operation_sync(error_handler):
    """Test synchronous retry operation."""
    # Mock function that fails twice then succeeds
    attempt = 0
    def flaky_function():
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            raise ConnectionError(f"Failed attempt {attempt}")
        return "Success"
    
    # Set up retry policy
    error_handler.set_retry_policy("test_component", RetryPolicy(
        max_retries=3,
        initial_delay_ms=50,  # Short delay for tests
        backoff_factor=1.0
    ))
    
    # Execute with retry
    result, error_report = error_handler.retry_operation_sync(
        flaky_function,
        context={"component": "test_component", "operation": "flaky_operation"}
    )
    
    # Check result
    assert result == "Success"
    assert error_report is None
    assert attempt == 3  # Function should have been called 3 times


def test_error_hooks(error_handler, mock_exception):
    """Test that error hooks are triggered correctly."""
    # Create a mock hook
    hook_called = False
    def test_hook(error_report):
        nonlocal hook_called
        hook_called = True
        assert error_report.error_type == ErrorType.VALIDATION
    
    # Register the hook
    error_handler.register_error_hook(ErrorType.VALIDATION, test_hook)
    
    # Create an error that should trigger the hook
    error_handler.create_error_report(mock_exception, {"component": "test"})
    
    # Check that the hook was called
    assert hook_called is True


def test_error_metrics(error_handler):
    """Test error metrics collection and reporting."""
    # Create some errors with different types
    error_handler.create_error_report(ValueError("Error 1"), {"component": "test"})
    error_handler.create_error_report(ConnectionError("Error 2"), {"component": "test"})
    error_handler.create_error_report(TimeoutError("Error 3"), {"component": "test"})
    error_handler.create_error_report(ValueError("Error 4"), {"component": "test"})
    
    # Get metrics
    metrics = error_handler.get_error_metrics()
    
    # Check error counts
    assert metrics["error_counts"]["validation"] == 2
    assert metrics["error_counts"]["network"] == 1
    assert metrics["error_counts"]["timeout"] == 1
    assert metrics["total_errors"] == 4


def test_safe_execute(error_handler):
    """Test safe execute wrapper."""
    # Function that works
    def success_func(x, y):
        return x + y
    
    # Function that raises
    def error_func(x, y):
        raise ValueError(f"Error with {x} and {y}")
    
    # Test with successful function
    result = safe_execute(success_func, 2, 3, error_handler=error_handler)
    assert result == 5
    
    # Test with failing function
    result = safe_execute(error_func, 2, 3, error_handler=error_handler, component="test")
    assert result is None
    
    # Check that error was handled
    errors = error_handler.get_unresolved_errors()
    assert len(errors) > 0
    assert any(e.message == "Error with 2 and 3" for e in errors)


@pytest.mark.asyncio
async def test_safe_execute_async(error_handler):
    """Test safe execute async wrapper."""
    # Async function that works
    async def success_func(x, y):
        await anyio.sleep(0.01)
        return x + y
    
    # Async function that raises
    async def error_func(x, y):
        await anyio.sleep(0.01)
        raise ValueError(f"Async error with {x} and {y}")
    
    # Test with successful function
    result = await safe_execute_async(success_func, 2, 3, error_handler=error_handler)
    assert result == 5
    
    # Test with failing function
    result = await safe_execute_async(error_func, 2, 3, error_handler=error_handler, component="test")
    assert result is None
    
    # Check that error was handled
    errors = error_handler.get_unresolved_errors()
    assert len(errors) > 0
    assert any(e.message == "Async error with 2 and 3" for e in errors)


def test_resolve_error(error_handler, mock_exception):
    """Test error resolution."""
    # Create an error
    error_report = error_handler.create_error_report(mock_exception, {"component": "test"})
    error_id = error_report.error_id
    
    # Resolve the error
    resolved = error_handler.resolve_error(error_id, "Manual fix", {"notes": "Fixed by test"})
    
    # Check resolution
    assert resolved is True
    updated_report = error_handler.get_error_report(error_id)
    assert updated_report.resolution_status == "resolved"
    assert updated_report.resolution_strategy == "Manual fix"
    assert updated_report.recovery_details["manual_resolution"]["details"]["notes"] == "Fixed by test"


def test_error_history(error_handler):
    """Test error history filtering."""
    # Create errors with different timestamps manually
    # First error 10 hours ago
    ten_hours_ago = datetime.now() - timedelta(hours=10)
    error1 = error_handler.create_error_report(ValueError("Old error"), {"component": "comp1"})
    error1.context.timestamp = ten_hours_ago
    
    # Second error 5 hours ago
    five_hours_ago = datetime.now() - timedelta(hours=5)
    error2 = error_handler.create_error_report(ValueError("Medium error"), {"component": "comp2"})
    error2.context.timestamp = five_hours_ago
    
    # Third error 1 hour ago
    one_hour_ago = datetime.now() - timedelta(hours=1)
    error3 = error_handler.create_error_report(ValueError("Recent error"), {"component": "comp1"})
    error3.context.timestamp = one_hour_ago
    
    # Get history for the last 6 hours
    history = error_handler.get_error_history(period_hours=6)
    
    # Should include the second and third errors but not the first
    validation_errors = history.get(ErrorType.VALIDATION.value, [])
    assert any(e.error_id == error2.error_id for e in validation_errors)
    assert any(e.error_id == error3.error_id for e in validation_errors)
    assert not any(e.error_id == error1.error_id for e in validation_errors)
    
    # Filter by component
    comp1_history = error_handler.get_error_history(component="comp1")
    comp1_errors = comp1_history.get(ErrorType.VALIDATION.value, [])
    assert any(e.error_id == error1.error_id for e in comp1_errors)
    assert any(e.error_id == error3.error_id for e in comp1_errors)
    assert not any(e.error_id == error2.error_id for e in comp1_errors)


def test_integration_with_execution_orchestrator(error_handler):
    """Test integration with execution orchestrator."""
    # Mock execution orchestrator
    mock_orchestrator = Mock()
    mock_orchestrator.set_test_end_hook = Mock()
    
    # Integrate
    error_handler.integrate_with_execution_orchestrator(mock_orchestrator)
    
    # Verify that hook was set
    assert mock_orchestrator.set_test_end_hook.called


def test_integration_with_dependency_manager(error_handler):
    """Test integration with dependency manager."""
    # Mock dependency manager with validate_dependencies method
    mock_dependency_manager = Mock()
    original_validate = Mock(return_value=(True, []))
    mock_dependency_manager.validate_dependencies = original_validate
    
    # Integrate
    error_handler.integrate_with_dependency_manager(mock_dependency_manager)
    
    # Verify that the method was replaced
    assert mock_dependency_manager.validate_dependencies != original_validate
    
    # Call the new method to test
    result = mock_dependency_manager.validate_dependencies()
    assert result == (True, [])
    assert original_validate.called


def test_clear_resolved_errors(error_handler):
    """Test clearing resolved errors."""
    # Create some errors
    error1 = error_handler.create_error_report(ValueError("Error 1"), {"component": "test"})
    error2 = error_handler.create_error_report(ValueError("Error 2"), {"component": "test"})
    error3 = error_handler.create_error_report(ValueError("Error 3"), {"component": "test"})
    
    # Resolve two of them
    error_handler.resolve_error(error1.error_id, "Fixed")
    error_handler.resolve_error(error2.error_id, "Fixed")
    
    # Clear resolved errors
    cleared = error_handler.clear_resolved_errors()
    
    # Check that only resolved errors were cleared
    assert cleared == 2
    assert error1.error_id not in error_handler.errors
    assert error2.error_id not in error_handler.errors
    assert error3.error_id in error_handler.errors


if __name__ == "__main__":
    pytest.main(["-v", __file__])