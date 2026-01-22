#!/usr/bin/env python3
"""
Test script for Enhanced Error Recovery and Performance-Based Recovery System

This script tests the enhanced error recovery and performance-based recovery system
for the Distributed Testing Framework.

Usage:
    python run_test_enhanced_error_recovery.py
"""

import asyncio
import logging
import json
import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_error_recovery")

# Import required modules
try:
    from distributed_error_handler import DistributedErrorHandler, ErrorType, ErrorSeverity
    from error_recovery_strategies import EnhancedErrorRecoveryManager
    from error_recovery_with_performance_tracking import PerformanceBasedErrorRecovery
    from enhanced_error_handling_integration import install_enhanced_error_handling
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure you're running this script from the distributed_testing directory or the PYTHONPATH is set correctly.")
    sys.exit(1)


class MockTask:
    """Mock task for testing."""
    
    def __init__(self, task_id, status="pending"):
        self.task_id = task_id
        self.status = status
        self.parameters = {"test": True}
        self.created_at = datetime.now()
        self.retried_task_id = None


class MockCoordinator:
    """Mock coordinator for testing."""
    
    def __init__(self, with_db=True):
        """Initialize mock coordinator."""
        self.tasks = {}
        self.running_tasks = {}
        self.pending_tasks = set()
        self.worker_connections = {}
        self.workers = {}
        
        if with_db:
            try:
                import duckdb
                self.db = duckdb.connect(":memory:")
                logger.info("Using in-memory DuckDB database")
            except ImportError:
                logger.warning("DuckDB not available, running without database")
                self.db = None
        else:
            self.db = None
        
        # Add some mock tasks
        for i in range(10):
            task_id = f"task-{i}"
            status = "pending" if i < 3 else "running" if i < 8 else "completed"
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": status,
                "parameters": {"test": True},
                "created_at": datetime.now()
            }
            
            if status == "running":
                self.running_tasks[task_id] = f"worker-{i % 3}"
            elif status == "pending":
                self.pending_tasks.add(task_id)


class ErrorSimulator:
    """Simulates various error scenarios for testing."""
    
    def __init__(self, coordinator, error_handler):
        """Initialize error simulator."""
        self.coordinator = coordinator
        self.error_handler = error_handler
    
    def network_error(self):
        """Simulate a network error."""
        return ConnectionError("Connection refused when connecting to worker")
    
    def database_error(self):
        """Simulate a database error."""
        return Exception("Database connection error: failed to execute query")
    
    def worker_error(self):
        """Simulate a worker error."""
        return Exception("Worker went offline unexpectedly")
    
    def timeout_error(self):
        """Simulate a timeout error."""
        return TimeoutError("Operation timed out after 30 seconds")
    
    def system_error(self):
        """Simulate a system resource error."""
        return MemoryError("Out of memory when allocating tensor")
    
    def get_error_context(self, error_type):
        """Get context for an error."""
        contexts = {
            "network": {
                "component": "network",
                "operation": "connect_worker",
                "error_type": ErrorType.NETWORK,
                "error_severity": ErrorSeverity.MEDIUM,
                "related_entities": {
                    "worker_id": "worker-1"
                }
            },
            "database": {
                "component": "database",
                "operation": "execute_query",
                "error_type": ErrorType.DATABASE,
                "error_severity": ErrorSeverity.HIGH,
                "related_entities": {
                    "query": "SELECT * FROM tasks"
                }
            },
            "worker": {
                "component": "worker",
                "operation": "execute_task",
                "error_type": ErrorType.COORDINATION,
                "error_severity": ErrorSeverity.MEDIUM,
                "related_entities": {
                    "worker_id": "worker-2",
                    "task_id": "task-3"
                }
            },
            "timeout": {
                "component": "coordinator",
                "operation": "wait_response",
                "error_type": ErrorType.TIMEOUT,
                "error_severity": ErrorSeverity.MEDIUM
            },
            "system": {
                "component": "system",
                "operation": "allocate_resource",
                "error_type": ErrorType.RESOURCE,
                "error_severity": ErrorSeverity.HIGH
            }
        }
        
        return contexts.get(error_type, {
            "component": "unknown",
            "operation": "unknown",
            "error_type": ErrorType.UNKNOWN,
            "error_severity": ErrorSeverity.MEDIUM
        })
    
    def generate_error(self, error_type):
        """Generate an error of the specified type."""
        error_generators = {
            "network": self.network_error,
            "database": self.database_error,
            "worker": self.worker_error,
            "timeout": self.timeout_error,
            "system": self.system_error
        }
        
        generator = error_generators.get(error_type, self.network_error)
        context = self.get_error_context(error_type)
        
        return generator(), context


async def test_handle_error(error_handler, error_type):
    """Test error handling for a specific error type."""
    simulator = ErrorSimulator(error_handler.coordinator, error_handler)
    
    logger.info(f"Testing error handling for {error_type} error")
    
    # Generate error
    error, context = simulator.generate_error(error_type)
    
    # Handle error
    start_time = time.time()
    success, info = await error_handler.handle_error(error, context)
    execution_time = time.time() - start_time
    
    # Print results
    print(f"\nError Type: {error_type}")
    print(f"Recovery Strategy: {info['strategy_name']}")
    print(f"Success: {success}")
    print(f"Recovery Level: {info['recovery_level']}")
    print(f"Execution Time: {info['execution_time']:.2f} seconds")
    print(f"Affected Tasks: {info.get('affected_tasks', 0)}")
    print(f"Recovered Tasks: {info.get('recovered_tasks', 0)}")
    
    # Return results for verification
    return {
        "error_type": error_type,
        "success": success,
        "strategy": info['strategy_name'],
        "recovery_level": info['recovery_level'],
        "execution_time": info['execution_time'],
        "affected_tasks": info.get('affected_tasks', 0),
        "recovered_tasks": info.get('recovered_tasks', 0)
    }


async def test_persistent_error(error_handler, error_type, attempts=3):
    """Test persistent error handling with progressive recovery."""
    simulator = ErrorSimulator(error_handler.coordinator, error_handler)
    
    logger.info(f"Testing progressive recovery for persistent {error_type} error")
    
    results = []
    error, context = simulator.generate_error(error_type)
    
    # Force recovery to fail by patching the strategies
    for strategy in error_handler.performance_recovery.recovery_manager.strategies.values():
        if hasattr(strategy, 'execute'):
            original_execute = strategy.execute
            
            # Make it fail for the first N-1 attempts
            # This function helps us create a closure with the right variables
            def create_patched_execute(original_fn, attempt_counter=[0]):
                async def patched_execute(error_info):
                    attempt_counter[0] += 1
                    if attempt_counter[0] >= attempts:
                        # Last attempt succeeds
                        return True
                    else:
                        # Earlier attempts fail
                        return False
                return patched_execute
            
            strategy.execute = create_patched_execute(original_execute)
    
    # Try multiple recovery attempts
    for i in range(attempts):
        logger.info(f"Attempt {i+1} of {attempts}")
        
        # Handle error
        success, info = await error_handler.handle_error(error, context)
        
        # Store results
        results.append({
            "attempt": i+1,
            "success": success,
            "strategy": info['strategy_name'],
            "recovery_level": info['recovery_level'],
            "execution_time": info['execution_time']
        })
        
        # Print results
        print(f"\nAttempt {i+1}:")
        print(f"Recovery Strategy: {info['strategy_name']}")
        print(f"Success: {success}")
        print(f"Recovery Level: {info['recovery_level']}")
        print(f"Execution Time: {info['execution_time']:.2f} seconds")
        
        # Break if successful
        if success:
            logger.info(f"Recovery succeeded on attempt {i+1}")
            break
    
    # Verify that recovery level escalated
    if len(results) > 1:
        initial_level = results[0]["recovery_level"]
        final_level = results[-1]["recovery_level"]
        print(f"\nRecovery level escalation: {initial_level} -> {final_level}")
    
    # Get recovery history
    if len(results) > 0:
        # Get recovery history for the first error
        error_id = None
        for error_id in error_handler.performance_recovery.error_recovery_levels:
            # Use the first one we find
            break
        
        if error_id:
            history = error_handler.get_recovery_history(error_id)
            print("\nRecovery History:")
            print(json.dumps(history, indent=2))
    
    return results


async def test_metrics_and_diagnostics(error_handler):
    """Test metrics and diagnostics."""
    logger.info("Testing metrics and diagnostics")
    
    # Get performance metrics
    metrics = error_handler.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"Total Executions: {metrics['overall']['total_executions']}")
    print(f"Strategy Count: {metrics['overall']['strategy_count']}")
    print(f"Error Type Count: {metrics['overall']['error_type_count']}")
    
    # Get error metrics
    error_metrics = error_handler.get_error_metrics()
    print("\nError Metrics:")
    print(f"Total Errors: {error_metrics.get('total_errors', 0)}")
    print(f"Unresolved Errors: {error_metrics.get('unresolved_errors', 0)}")
    
    # Run diagnostics
    diagnostics = await error_handler.run_diagnostics()
    print("\nDiagnostics:")
    print(f"Status: {diagnostics['status']}")
    if diagnostics['issues']:
        print("Issues:")
        for issue in diagnostics['issues']:
            print(f"- {issue}")
    else:
        print("No issues detected")
    
    return {
        "performance_metrics": metrics,
        "error_metrics": error_metrics,
        "diagnostics": diagnostics
    }


async def test_strategy_recommendations(error_handler):
    """Test strategy recommendations."""
    logger.info("Testing strategy recommendations")
    
    # Test for a few error types
    error_types = ["network", "database", "worker", "timeout", "system"]
    
    for error_type in error_types:
        recommendations = error_handler.get_strategy_recommendations(error_type)
        
        print(f"\nRecommendations for {error_type} errors:")
        if recommendations:
            for i, rec in enumerate(recommendations[:3], 1):  # Top 3
                print(f"{i}. {rec['strategy_name']} (Score: {rec['score']:.2f})")
        else:
            print("No recommendations available")
    
    return True


async def test_error_resolution(error_handler):
    """Test error resolution."""
    logger.info("Testing error resolution")
    
    # Generate an error to resolve
    simulator = ErrorSimulator(error_handler.coordinator, error_handler)
    error, context = simulator.generate_error("network")
    
    # Handle error to generate an error report
    await error_handler.handle_error(error, context)
    
    # Get an error ID to resolve
    error_id = None
    for id in error_handler.error_handler.errors:
        error_id = id
        break
    
    if not error_id:
        print("No errors to resolve")
        return False
    
    # Resolve the error
    resolved = error_handler.resolve_error(
        error_id, 
        "Manual resolution for testing", 
        {"test": True}
    )
    
    print(f"\nResolved Error: {error_id}")
    print(f"Resolution Status: {'Success' if resolved else 'Failed'}")
    
    # Verify recovery level was reset
    level_reset = error_id not in error_handler.performance_recovery.error_recovery_levels
    print(f"Recovery Level Reset: {level_reset}")
    
    return {
        "error_id": error_id,
        "resolved": resolved,
        "level_reset": level_reset
    }


async def test_reset_recovery_levels(error_handler):
    """Test resetting recovery levels."""
    logger.info("Testing reset recovery levels")
    
    # Generate a few errors
    simulator = ErrorSimulator(error_handler.coordinator, error_handler)
    error_types = ["network", "database", "worker"]
    
    for error_type in error_types:
        error, context = simulator.generate_error(error_type)
        await error_handler.handle_error(error, context)
    
    # Set some recovery levels
    levels_before = len(error_handler.performance_recovery.error_recovery_levels)
    print(f"\nRecovery Levels Before: {levels_before}")
    
    # Reset all recovery levels
    reset = error_handler.reset_all_recovery_levels()
    
    # Check levels after reset
    levels_after = len(error_handler.performance_recovery.error_recovery_levels)
    print(f"Recovery Levels After: {levels_after}")
    print(f"Reset Success: {reset}")
    
    return {
        "levels_before": levels_before,
        "levels_after": levels_after,
        "reset_success": reset
    }


async def run_all_tests(coordinator=None, with_db=True):
    """Run all error handling integration tests."""
    # Create coordinator if not provided
    if coordinator is None:
        coordinator = MockCoordinator(with_db=with_db)
    
    # Install enhanced error handling
    error_handler = install_enhanced_error_handling(coordinator)
    
    results = {}
    
    # Test basic error handling for various error types
    results["basic_error_handling"] = {}
    for error_type in ["network", "database", "worker", "timeout", "system"]:
        results["basic_error_handling"][error_type] = await test_handle_error(error_handler, error_type)
    
    # Test persistent error with progressive recovery
    results["progressive_recovery"] = await test_persistent_error(error_handler, "network", attempts=3)
    
    # Test metrics and diagnostics
    results["metrics_and_diagnostics"] = await test_metrics_and_diagnostics(error_handler)
    
    # Test strategy recommendations
    results["strategy_recommendations"] = await test_strategy_recommendations(error_handler)
    
    # Test error resolution
    results["error_resolution"] = await test_error_resolution(error_handler)
    
    # Test reset recovery levels
    results["reset_recovery_levels"] = await test_reset_recovery_levels(error_handler)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    print("\nBasic Error Handling:")
    for error_type, result in results["basic_error_handling"].items():
        success = result["success"]
        print(f"  {error_type}: {'SUCCESS' if success else 'FAILED'}")
    
    print("\nProgressive Recovery:")
    final_attempt = results["progressive_recovery"][-1]
    print(f"  Final Attempt: {'SUCCESS' if final_attempt['success'] else 'FAILED'}")
    print(f"  Recovery Level Escalation: {results['progressive_recovery'][0]['recovery_level']} -> {final_attempt['recovery_level']}")
    
    print("\nMetrics and Diagnostics:")
    diagnostics = results["metrics_and_diagnostics"]["diagnostics"]
    print(f"  Status: {diagnostics['status']}")
    
    print("\nError Resolution:")
    resolution = results["error_resolution"]
    print(f"  Success: {'SUCCESS' if resolution['resolved'] else 'FAILED'}")
    
    print("\nReset Recovery Levels:")
    reset = results["reset_recovery_levels"]
    print(f"  Success: {'SUCCESS' if reset['reset_success'] else 'FAILED'}")
    
    # Check if all critical tests passed
    all_basic_passed = all(result["success"] for result in results["basic_error_handling"].values())
    progressive_passed = results["progressive_recovery"][-1]["success"]
    resolution_passed = results["error_resolution"]["resolved"]
    reset_passed = results["reset_recovery_levels"]["reset_success"]
    
    all_passed = all_basic_passed and progressive_passed and resolution_passed and reset_passed
    
    print("\n" + "="*50)
    print(f"OVERALL RESULT: {'SUCCESS' if all_passed else 'FAILED'}")
    print("="*50)
    
    return results, all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Enhanced Error Recovery System")
    parser.add_argument("--no-db", action="store_true", help="Run without database")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", help="Output file for JSON results")
    args = parser.parse_args()
    
    # Run all tests
    results, success = asyncio.run(run_all_tests(with_db=not args.no_db))
    
    # Output JSON if requested
    if args.json:
        # Convert results to JSON-serializable format
        json_results = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "results": {}
        }
        
        # Clean up non-serializable objects
        for test_name, test_results in results.items():
            if isinstance(test_results, dict):
                json_results["results"][test_name] = {}
                for key, value in test_results.items():
                    # Handle nested dictionaries
                    if isinstance(value, dict):
                        json_results["results"][test_name][key] = {}
                        for inner_key, inner_value in value.items():
                            # Convert values to serializable types
                            if isinstance(inner_value, (dict, list, str, int, float, bool, type(None))):
                                json_results["results"][test_name][key][inner_key] = inner_value
                            else:
                                json_results["results"][test_name][key][inner_key] = str(inner_value)
                    else:
                        # Convert values to serializable types
                        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                            json_results["results"][test_name][key] = value
                        else:
                            json_results["results"][test_name][key] = str(value)
            elif isinstance(test_results, list):
                json_results["results"][test_name] = []
                for item in test_results:
                    if isinstance(item, dict):
                        cleaned_item = {}
                        for key, value in item.items():
                            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                                cleaned_item[key] = value
                            else:
                                cleaned_item[key] = str(value)
                        json_results["results"][test_name].append(cleaned_item)
                    else:
                        json_results["results"][test_name].append(str(item))
            else:
                json_results["results"][test_name] = str(test_results)
        
        # Output JSON results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(json_results, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(json_results, indent=2))
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)