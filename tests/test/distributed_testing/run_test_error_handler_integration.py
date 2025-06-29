#!/usr/bin/env python3
"""
Integration Example for Distributed Error Handler

This script demonstrates how to integrate the distributed error handler with
the test dependency manager and execution orchestrator.
"""

import asyncio
import random
import time
import uuid
import json
import logging
from datetime import datetime

# Import components
from test_dependency_manager import TestDependencyManager, Dependency, DependencyType
from execution_orchestrator import ExecutionOrchestrator, ExecutionStrategy
from distributed_error_handler import (
    DistributedErrorHandler, ErrorType, ErrorSeverity, RetryPolicy,
    safe_execute, safe_execute_async
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("error_handler_integration")


def create_test_dependency_graph(dependency_manager):
    """Create a test dependency graph for demonstration."""
    # Register some test cases with dependencies
    dependency_manager.register_test("test_1", [], ["group_a"])
    dependency_manager.register_test("test_2", [Dependency("test_1")], ["group_a"])
    dependency_manager.register_test("test_3", [Dependency("test_1"), Dependency("test_2")], ["group_b"])
    dependency_manager.register_test("test_4", [Dependency("group_a", is_group=True)], ["group_b"])
    dependency_manager.register_test("test_5", [
        Dependency("test_3"), 
        Dependency("test_4", dependency_type=DependencyType.SOFT)
    ])
    dependency_manager.register_test("test_6", [Dependency("test_5")])
    dependency_manager.register_test("test_7", [Dependency("test_1"), Dependency("test_6", dependency_type=DependencyType.OPTIONAL)])
    dependency_manager.register_test("test_8", [Dependency("test_3"), Dependency("test_7")])
    
    # Add some tests with circular dependencies to test error handling
    if random.random() > 0.7:  # Occasionally introduce circular dependencies to test error handling
        dependency_manager.register_test("error_test_1", [Dependency("error_test_2")])
        dependency_manager.register_test("error_test_2", [Dependency("error_test_1")])


def setup_error_hooks(error_handler):
    """Set up error hooks for different error types."""
    
    # Hook for test execution errors
    def test_execution_error_hook(error_report):
        logger.warning(f"Test execution error hook triggered: {error_report.message}")
        
        # You could implement recovery logic here, such as:
        # - Notifying administrators
        # - Updating a dashboard
        # - Attempting to clean up resources
        # - Recording metrics
    
    # Hook for dependency errors
    def dependency_error_hook(error_report):
        logger.warning(f"Dependency error hook triggered: {error_report.message}")
        
        # For dependency errors, you might want to:
        # - Update the dependency graph
        # - Try alternative resolution strategies
        # - Skip certain tests
    
    # Register hooks for different error types
    error_handler.register_error_hook(ErrorType.TEST_EXECUTION, test_execution_error_hook)
    error_handler.register_error_hook(ErrorType.DEPENDENCY, dependency_error_hook)
    error_handler.register_error_hook([ErrorType.CONFIGURATION, ErrorType.VALIDATION], 
                                   lambda e: logger.warning(f"Config/validation error: {e.message}"))


def setup_retry_policies(error_handler):
    """Set up retry policies for different components."""
    
    # Default retry policy (used if no component-specific policy is defined)
    error_handler.default_retry_policy = RetryPolicy(
        max_retries=3,
        initial_delay_ms=1000,
        backoff_factor=2.0,
        jitter=True
    )
    
    # Dependency manager-specific policy
    error_handler.set_retry_policy("dependency_manager", RetryPolicy(
        max_retries=2,
        initial_delay_ms=500,
        backoff_factor=1.5,
        skip_on_error_types=[ErrorType.VALIDATION]  # Don't retry validation errors
    ))
    
    # Test execution-specific policy
    error_handler.set_retry_policy("test_execution", RetryPolicy(
        max_retries=3,
        initial_delay_ms=2000,
        backoff_factor=2.0,
        skip_if_severity_above=ErrorSeverity.HIGH  # Don't retry critical errors
    ))


def setup_mock_database(error_handler):
    """Set up a mock database connection for the error handler."""
    
    # Create a mock database connection
    class MockDBConnection:
        def __init__(self):
            self.executed_queries = []
        
        def execute(self, query, params=None):
            self.executed_queries.append((query, params))
            return []
        
        def close(self):
            pass
    
    # Assign to error handler
    error_handler.db_connection = MockDBConnection()
    
    # Create schema
    error_handler._create_schema()


def add_execution_hooks(orchestrator, error_handler):
    """Add hooks to track execution and catch errors."""
    
    # Hook called before execution starts
    def pre_execution_hook(max_workers, strategy, total_tests):
        logger.info(f"Starting test execution with {max_workers} workers, " 
                   f"{strategy.name} strategy, and {total_tests} total tests")
    
    # Hook called after execution completes
    def post_execution_hook(metrics):
        logger.info(f"Execution completed: {metrics['completed_tests']} successful, "
                  f"{metrics['failed_tests']} failed, {metrics['skipped_tests']} skipped")
        
        # Log any errors that occurred
        unresolved_errors = error_handler.get_unresolved_errors()
        if unresolved_errors:
            logger.warning(f"There are {len(unresolved_errors)} unresolved errors")
            for error in unresolved_errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error.error_type.value}: {error.message}")
    
    # Hook called when a test starts
    def test_start_hook(test_id, worker_id):
        logger.debug(f"Test {test_id} started on worker {worker_id}")
    
    # Hook called when a test ends
    def test_end_hook(test_id, status, result, error):
        if error:
            logger.warning(f"Test {test_id} failed with error: {error}")
            
            # Create an error report
            try:
                # Simulate an exception to capture stack trace
                raise Exception(error)
            except Exception as e:
                error_handler.handle_error(e, {
                    "component": "test_execution",
                    "operation": f"execute_test:{test_id}",
                    "related_entities": {"test_id": test_id, "worker_id": result.get("worker_id") if result else None},
                    "metadata": {"status": str(status)}
                })
        else:
            logger.debug(f"Test {test_id} completed with status: {status}")
    
    # Register hooks
    orchestrator.set_pre_execution_hook(pre_execution_hook)
    orchestrator.set_post_execution_hook(post_execution_hook)
    orchestrator.set_test_start_hook(test_start_hook)
    orchestrator.set_test_end_hook(test_end_hook)


async def run_integration_demo():
    """Run the integration demo."""
    # Create components
    dependency_manager = TestDependencyManager()
    error_handler = DistributedErrorHandler()
    
    # Set up mock database for error persistence
    setup_mock_database(error_handler)
    
    # Create simple test dependency graph
    dependency_manager.register_test("test_1", [])
    dependency_manager.register_test("test_2", [Dependency("test_1")])
    
    # Set up error hooks
    setup_error_hooks(error_handler)
    
    # Set up retry policies
    setup_retry_policies(error_handler)
    
    # Integrate error handler with dependency manager
    error_handler.integrate_with_dependency_manager(dependency_manager)
    
    # Demo of retry with dependency validation
    logger.info("\n=== Demo: Dependency Validation with Retry ===")
    try:
        # Validate dependencies (this will use the wrapped version from integration)
        is_valid, errors = dependency_manager.validate_dependencies()
        if not is_valid:
            logger.error(f"Dependency validation failed: {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
        else:
            logger.info("Dependency validation successful")
    except Exception as e:
        logger.error(f"Error during dependency validation: {str(e)}")
    
    # Test error creation and handling
    logger.info("\n=== Demo: Error Handling and Retry ===")
    
    # Create some test errors
    error_handler.create_error_report(ValueError("Test error 1"), {"component": "test"})
    error_handler.create_error_report(ConnectionError("Connection failed"), {"component": "network"})
    
    # Demonstrate retry
    retry_success = False
    
    # Function that fails twice then succeeds
    attempt = 0
    def flaky_func():
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            raise ConnectionError(f"Connection failed (attempt {attempt})")
        return "Success on attempt " + str(attempt)
    
    # Execute with retry
    result, error = await error_handler.retry_operation(
        flaky_func,
        context={"component": "demo", "operation": "flaky_operation"}
    )
    
    if result:
        logger.info(f"Retry successful: {result}")
    else:
        logger.error(f"Retry failed: {error.message}")
    
    # Demo of error metrics and reporting
    logger.info("\n=== Demo: Error Metrics and Reporting ===")
    
    # Get error metrics
    metrics = error_handler.get_error_metrics()
    logger.info(f"Total errors: {metrics['total_errors']}")
    logger.info(f"Error types: {metrics['error_counts']}")
    
    # Get unresolved errors
    unresolved = error_handler.get_unresolved_errors()
    logger.info(f"Unresolved errors: {len(unresolved)}")
    
    # Resolve errors
    if unresolved:
        # Resolve the first error
        error = unresolved[0]
        resolved = error_handler.resolve_error(
            error.error_id, 
            "Manual resolution",
            {"notes": "Resolved during demo"}
        )
        if resolved:
            logger.info(f"Resolved error {error.error_id}")
        
        # Clear resolved errors
        cleared = error_handler.clear_resolved_errors()
        logger.info(f"Cleared {cleared} resolved errors from memory")
    
    logger.info("\nIntegration demo completed successfully")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(run_integration_demo())