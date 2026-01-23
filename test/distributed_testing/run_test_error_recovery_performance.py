#!/usr/bin/env python3
"""
Distributed Testing Framework - Error Recovery with Performance Tracking Demo

This script demonstrates the Error Recovery with Performance Tracking capabilities
of the Distributed Testing Framework.
"""

import anyio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
import duckdb
import random
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("error_recovery_demo")

# Import related modules
from distributed_error_handler import (
    DistributedErrorHandler, ErrorType, ErrorSeverity, 
    ErrorContext, ErrorReport
)

from error_recovery_strategies import (
    ErrorCategory, RecoveryStrategy, RetryStrategy, WorkerRecoveryStrategy,
    DatabaseRecoveryStrategy, CoordinatorRecoveryStrategy, SystemRecoveryStrategy,
    EnhancedErrorRecoveryManager
)

from error_recovery_with_performance import (
    ErrorRecoveryWithPerformance, RecoveryPerformanceMetric,
    RecoveryPerformanceRecord, RecoveryStrategyScore,
    ProgressiveRecoveryLevel
)


class DemoCoordinator:
    """A demo coordinator for testing error recovery."""
    
    def __init__(self, db_path=":memory:"):
        """Initialize the demo coordinator."""
        # Initialize coordinator components
        self.worker_connections = {}
        self.tasks = {}
        self.running_tasks = {}
        self.pending_tasks = set()
        self.failed_tasks = set()
        self.workers = {}
        
        # Connect to database
        self.db_path = db_path
        self.db = duckdb.connect(db_path)
        
        # Create demo schema
        self._create_schema()
        
        logger.info("Demo coordinator initialized")
    
    def _create_schema(self):
        """Create database schema."""
        try:
            # Create tables
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id VARCHAR PRIMARY KEY,
                hostname VARCHAR,
                registration_time TIMESTAMP,
                last_heartbeat TIMESTAMP,
                status VARCHAR,
                capabilities JSON
            )
            """)
            
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id VARCHAR PRIMARY KEY,
                type VARCHAR,
                status VARCHAR,
                created TIMESTAMP,
                started TIMESTAMP,
                ended TIMESTAMP,
                worker_id VARCHAR,
                attempts INTEGER,
                config JSON
            )
            """)
            
            logger.info("Database schema created")
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
    
    def add_worker(self, worker_id, hostname="localhost"):
        """Add a worker to the coordinator."""
        # Create worker record
        worker = {
            "worker_id": worker_id,
            "hostname": hostname,
            "registration_time": datetime.now(),
            "last_heartbeat": datetime.now(),
            "status": "active",
            "capabilities": {"cpu": True, "memory": 8192, "gpu": False}
        }
        
        # Store in memory
        self.workers[worker_id] = worker
        
        # Mock connection
        self.worker_connections[worker_id] = MockWebSocket(open=True)
        
        # Store in database
        self.db.execute("""
        INSERT INTO workers (
            worker_id, hostname, registration_time, last_heartbeat, status, capabilities
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            worker_id,
            hostname,
            worker["registration_time"],
            worker["last_heartbeat"],
            worker["status"],
            json.dumps(worker["capabilities"])
        ))
        
        logger.info(f"Added worker {worker_id}")
        return worker
    
    def add_task(self, task_id, task_type="test", status="pending"):
        """Add a task to the coordinator."""
        # Create task record
        task = {
            "task_id": task_id,
            "type": task_type,
            "status": status,
            "created": datetime.now(),
            "started": None,
            "ended": None,
            "worker_id": None,
            "attempts": 0,
            "config": {
                "retry_policy": {
                    "max_retries": 3,
                    "backoff_factor": 2.0
                }
            }
        }
        
        # Store in memory
        self.tasks[task_id] = task
        
        # Add to pending tasks if pending
        if status == "pending":
            self.pending_tasks.add(task_id)
        
        # Store in database
        self.db.execute("""
        INSERT INTO tasks (
            task_id, type, status, created, started, ended,
            worker_id, attempts, config
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            task["type"],
            task["status"],
            task["created"],
            task["started"],
            task["ended"],
            task["worker_id"],
            task["attempts"],
            json.dumps(task["config"])
        ))
        
        logger.info(f"Added task {task_id}")
        return task
    
    def assign_task(self, task_id, worker_id):
        """Assign a task to a worker."""
        # Check task exists
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        # Check worker exists
        if worker_id not in self.workers:
            logger.error(f"Worker {worker_id} not found")
            return False
        
        # Update task
        task = self.tasks[task_id]
        task["status"] = "running"
        task["started"] = datetime.now()
        task["worker_id"] = worker_id
        
        # Update running tasks
        self.running_tasks[task_id] = worker_id
        
        # Remove from pending tasks
        if task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)
        
        # Update database
        self.db.execute("""
        UPDATE tasks
        SET status = ?, started = ?, worker_id = ?
        WHERE task_id = ?
        """, (
            task["status"],
            task["started"],
            task["worker_id"],
            task_id
        ))
        
        logger.info(f"Assigned task {task_id} to worker {worker_id}")
        return True
    
    async def _assign_pending_tasks(self):
        """Assign pending tasks to available workers."""
        assigned_count = 0
        
        # Get active workers
        active_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker["status"] == "active"
        ]
        
        if not active_workers:
            logger.warning("No active workers available")
            return 0
        
        # Assign tasks
        for task_id in list(self.pending_tasks):
            # Skip if no more active workers
            if not active_workers:
                break
            
            # Assign to a random worker
            worker_id = random.choice(active_workers)
            if self.assign_task(task_id, worker_id):
                assigned_count += 1
        
        logger.info(f"Assigned {assigned_count} pending tasks")
        return assigned_count


class MockWebSocket:
    """A mock WebSocket for testing."""
    
    def __init__(self, open=True):
        """Initialize the mock WebSocket."""
        self.closed = not open
    
    async def close(self):
        """Close the WebSocket."""
        self.closed = True


class ErrorSimulator:
    """Simulates errors for testing recovery."""
    
    def __init__(self, error_handler, coordinator):
        """Initialize the error simulator."""
        self.error_handler = error_handler
        self.coordinator = coordinator
        
        # Error templates
        self.error_templates = {
            "worker_offline": self._create_worker_offline_error,
            "task_timeout": self._create_task_timeout_error,
            "database_connection": self._create_database_error,
            "coordinator_state": self._create_coordinator_error,
            "memory_exhaustion": self._create_resource_error,
        }
        
        logger.info("Error simulator initialized")
    
    def _create_worker_offline_error(self, worker_id):
        """Create a worker offline error."""
        if worker_id not in self.coordinator.workers:
            worker_id = list(self.coordinator.workers.keys())[0] if self.coordinator.workers else "worker-1"
        
        # Create error
        exception = ConnectionError(f"Worker {worker_id} connection lost")
        context = {
            "component": "worker",
            "operation": "heartbeat",
            "related_entities": {
                "worker_id": worker_id
            }
        }
        
        return exception, context
    
    def _create_task_timeout_error(self, task_id):
        """Create a task timeout error."""
        if task_id not in self.coordinator.tasks:
            task_id = list(self.coordinator.tasks.keys())[0] if self.coordinator.tasks else "task-1"
        
        # Get worker if assigned
        worker_id = None
        if task_id in self.coordinator.running_tasks:
            worker_id = self.coordinator.running_tasks[task_id]
        
        # Create error
        exception = asyncio.TimeoutError(f"Task {task_id} execution timed out")
        context = {
            "component": "task_execution",
            "operation": "execute",
            "related_entities": {
                "task_id": task_id,
                "worker_id": worker_id
            }
        }
        
        return exception, context
    
    def _create_database_error(self, db_operation="query"):
        """Create a database error."""
        # Create error
        exception = duckdb.Error(f"Database {db_operation} error: connection lost")
        context = {
            "component": "database",
            "operation": db_operation,
            "critical": db_operation in ["schema", "initialize"]
        }
        
        return exception, context
    
    def _create_coordinator_error(self, operation="state_sync"):
        """Create a coordinator error."""
        # Create error
        exception = RuntimeError(f"Coordinator state error: {operation} failed")
        context = {
            "component": "coordinator",
            "operation": operation,
            "critical": operation in ["initialize", "state_sync"]
        }
        
        return exception, context
    
    def _create_resource_error(self, resource_type="memory"):
        """Create a resource error."""
        # Create error
        exception = MemoryError(f"Resource exhaustion: {resource_type} limit exceeded")
        context = {
            "component": "system",
            "operation": "allocate_resource",
            "related_entities": {
                "resource_type": resource_type
            },
            "critical": resource_type in ["memory", "disk"]
        }
        
        return exception, context
    
    def simulate_error(self, error_type, **kwargs):
        """Simulate an error and create an error report."""
        # Get error template
        template_func = self.error_templates.get(error_type)
        if not template_func:
            logger.error(f"Unknown error type: {error_type}")
            return None
        
        # Create error
        exception, context = template_func(**kwargs)
        
        # Create error report
        error_report = self.error_handler.create_error_report(exception, context)
        
        logger.info(f"Simulated error: {error_type} - {error_report.error_id}")
        return error_report


class RecoveryAnalyzer:
    """Analyzes recovery performance."""
    
    def __init__(self, recovery_system):
        """Initialize the recovery analyzer."""
        self.recovery_system = recovery_system
    
    def print_performance_metrics(self, strategy_id=None, error_type=None):
        """Print performance metrics."""
        metrics = self.recovery_system.get_performance_metrics(strategy_id, error_type)
        
        print("\n===== RECOVERY PERFORMANCE METRICS =====")
        print(f"Total strategies: {metrics['summary']['total_strategies']}")
        print(f"Total error types: {metrics['summary']['total_error_types']}")
        print(f"Average success rate: {metrics['summary']['average_success_rate']:.2f}")
        print(f"Average recovery time: {metrics['summary']['average_recovery_time']:.2f} seconds")
        
        if metrics['strategy_stats']:
            print("\n----- Strategy Statistics -----")
            for sid, stats in metrics['strategy_stats'].items():
                print(f"Strategy: {stats['name']} ({sid})")
                print(f"  Total samples: {stats['total_samples']}")
                print(f"  Success rate: {stats['success_rate']:.2f}")
                print(f"  Avg recovery time: {stats['avg_recovery_time']:.2f} seconds")
                print(f"  Overall score: {stats['overall_score']:.2f}")
                print()
        
        if metrics['top_strategies']:
            print("\n----- Top Strategies by Error Type -----")
            for et, top in metrics['top_strategies'].items():
                print(f"Error type: {et}")
                print(f"  Best strategy: {top['strategy_name']} ({top['strategy_id']})")
                print(f"  Score: {top['score']:.2f}")
                print()
    
    def print_progressive_recovery_summary(self, error_id=None):
        """Print progressive recovery summary."""
        if error_id:
            # Print for specific error
            history = self.recovery_system.get_progressive_recovery_history(error_id)
            
            print(f"\n===== PROGRESSIVE RECOVERY FOR ERROR {error_id} =====")
            print(f"Current level: {history['current_level']}")
            print(f"History entries: {len(history['history'])}")
            
            for entry in history['history']:
                print(f"  Level: {entry['old_level']} -> {entry['new_level']}")
                print(f"  Strategy: {entry['strategy_name']} ({entry['strategy_id']})")
                print(f"  Success: {entry['success']}")
                print(f"  Timestamp: {entry['timestamp']}")
                if 'details' in entry and entry['details']:
                    print(f"  Details: {entry['details']}")
                print()
        else:
            # Print summary for all errors
            summary = self.recovery_system.get_progressive_recovery_history()
            
            print("\n===== PROGRESSIVE RECOVERY SUMMARY =====")
            print(f"Total errors: {len(summary['errors'])}")
            print(f"Level 1 errors: {summary['summary']['level_1_count']}")
            print(f"Level 2 errors: {summary['summary']['level_2_count']}")
            print(f"Level 3 errors: {summary['summary']['level_3_count']}")
            print(f"Level 4 errors: {summary['summary']['level_4_count']}")
            print(f"Level 5 errors: {summary['summary']['level_5_count']}")
            print(f"Successful recoveries: {summary['summary']['successful_recoveries']}")
            print(f"Failed recoveries: {summary['summary']['failed_recoveries']}")
            
            if summary['errors']:
                print("\n----- Error Details -----")
                for error in summary['errors']:
                    print(f"Error: {error['error_id']}")
                    print(f"  Current level: {error['current_level']}")
                    print(f"  Attempts: {error['attempts']}")
                    print(f"  Last attempt success: {error['last_attempt_success']}")
                    print(f"  Last attempt time: {error['last_attempt_time']}")
                    print()
    
    def print_adaptive_timeouts(self, error_type=None, strategy_id=None):
        """Print adaptive timeouts."""
        timeouts = self.recovery_system.get_adaptive_timeouts(error_type, strategy_id)
        
        print("\n===== ADAPTIVE TIMEOUTS =====")
        if not timeouts:
            print("No adaptive timeouts configured yet.")
            return
        
        for key, timeout in timeouts.items():
            et, sid = key.split(":")
            print(f"Error type: {et}")
            print(f"Strategy: {sid}")
            print(f"Timeout: {timeout:.2f} seconds")
            print()


async def setup_test_environment(db_path=":memory:"):
    """Set up the test environment."""
    # Create coordinator
    coordinator = DemoCoordinator(db_path)
    
    # Add some workers
    for i in range(3):
        coordinator.add_worker(f"worker-{i+1}")
    
    # Add some tasks
    for i in range(5):
        coordinator.add_task(f"task-{i+1}")
    
    # Assign some tasks
    coordinator.assign_task("task-1", "worker-1")
    coordinator.assign_task("task-2", "worker-2")
    
    # Create error handler
    error_handler = DistributedErrorHandler()
    
    # Create recovery manager
    recovery_manager = EnhancedErrorRecoveryManager(coordinator)
    
    # Add recovery strategies
    recovery_manager.strategies = {
        "retry": RetryStrategy(coordinator),
        "worker": WorkerRecoveryStrategy(coordinator),
        "database": DatabaseRecoveryStrategy(coordinator),
        "coordinator": CoordinatorRecoveryStrategy(coordinator),
        "system": SystemRecoveryStrategy(coordinator)
    }
    
    # Create error recovery system
    recovery_system = ErrorRecoveryWithPerformance(
        error_handler=error_handler,
        recovery_manager=recovery_manager,
        coordinator=coordinator,
        db_connection=coordinator.db
    )
    
    # Create error simulator
    error_simulator = ErrorSimulator(error_handler, coordinator)
    
    # Create recovery analyzer
    recovery_analyzer = RecoveryAnalyzer(recovery_system)
    
    return coordinator, error_handler, recovery_system, error_simulator, recovery_analyzer


async def run_recovery_scenario(scenario_type, recovery_system, error_simulator, iterations=5):
    """Run a recovery scenario."""
    print(f"\n===== RUNNING SCENARIO: {scenario_type.upper()} =====")
    print(f"Iterations: {iterations}")
    
    results = {
        "successful": 0,
        "failed": 0,
        "recovery_info": []
    }
    
    for i in range(iterations):
        print(f"\n----- Iteration {i+1}/{iterations} -----")
        
        # Simulate error
        error_report = error_simulator.simulate_error(scenario_type)
        if not error_report:
            print(f"Failed to simulate {scenario_type} error")
            continue
        
        # Recover from error
        print(f"Recovering from error: {error_report.error_id}")
        start_time = time.time()
        success, info = await recovery_system.recover(error_report)
        end_time = time.time()
        
        # Record result
        if success:
            results["successful"] += 1
            print(f"Recovery successful in {end_time - start_time:.2f} seconds")
        else:
            results["failed"] += 1
            print(f"Recovery failed after {end_time - start_time:.2f} seconds")
        
        # Store info
        info["duration"] = end_time - start_time
        results["recovery_info"].append(info)
    
    # Print summary
    print(f"\n----- Scenario {scenario_type.upper()} Summary -----")
    print(f"Total iterations: {iterations}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['successful'] / iterations:.2%}")
    
    return results


async def run_progressive_recovery_test(recovery_system, error_simulator, error_type="worker_offline"):
    """Run a progressive recovery test."""
    print("\n===== RUNNING PROGRESSIVE RECOVERY TEST =====")
    
    # Simulate error
    error_report = error_simulator.simulate_error(error_type)
    if not error_report:
        print(f"Failed to simulate {error_type} error")
        return None
    
    # Initial recovery
    print(f"Initial recovery from error: {error_report.error_id}")
    
    # Force failure for demonstration
    original_execute = recovery_system.recovery_manager.strategies["retry"]._execute_impl
    
    # Mock implementation that fails first then succeeds
    async def mock_execute_impl(self, error_info):
        # Get current level
        level = recovery_system.error_recovery_levels.get(error_report.error_id, 1)
        
        # Fail at level 1-2, succeed at level 3+
        if level <= 2:
            print(f"Mocking failure at level {level}")
            return False
        else:
            print(f"Mocking success at level {level}")
            return True
    
    # Replace with mock
    recovery_system.recovery_manager.strategies["retry"]._execute_impl = mock_execute_impl.__get__(
        recovery_system.recovery_manager.strategies["retry"], 
        type(recovery_system.recovery_manager.strategies["retry"])
    )
    
    try:
        # Run recovery with progressive escalation
        print(f"Running recovery with progressive escalation...")
        success, info = await recovery_system.recover(error_report)
        
        # Check final result
        if success:
            print(f"Recovery eventually succeeded after progression")
            print(f"Final level: {recovery_system.error_recovery_levels.get(error_report.error_id, 1)}")
        else:
            print(f"Recovery failed even after progression")
            print(f"Final level: {recovery_system.error_recovery_levels.get(error_report.error_id, 1)}")
        
        return {
            "error_id": error_report.error_id,
            "success": success,
            "info": info
        }
    finally:
        # Restore original implementation
        recovery_system.recovery_manager.strategies["retry"]._execute_impl = original_execute


async def run_all_scenarios(recovery_system, error_simulator, iterations=3):
    """Run all recovery scenarios."""
    scenarios = [
        "worker_offline",
        "task_timeout",
        "database_connection",
        "coordinator_state",
        "memory_exhaustion"
    ]
    
    results = {}
    
    for scenario in scenarios:
        results[scenario] = await run_recovery_scenario(
            scenario, recovery_system, error_simulator, iterations
        )
    
    return results


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Error Recovery with Performance Tracking Demo")
    parser.add_argument("--db-path", default=":memory:", help="Path to database file (default: in-memory)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each scenario (default: 3)")
    parser.add_argument("--scenario", choices=["worker_offline", "task_timeout", "database_connection", "coordinator_state", "memory_exhaustion", "progressive", "all"], default="all", help="Scenario to run (default: all)")
    
    args = parser.parse_args()
    
    # Set up test environment
    print("Setting up test environment...")
    coordinator, error_handler, recovery_system, error_simulator, recovery_analyzer = await setup_test_environment(args.db_path)
    
    # Run scenarios
    if args.scenario == "all":
        results = await run_all_scenarios(recovery_system, error_simulator, args.iterations)
    elif args.scenario == "progressive":
        await run_progressive_recovery_test(recovery_system, error_simulator)
    else:
        results = await run_recovery_scenario(args.scenario, recovery_system, error_simulator, args.iterations)
    
    # Print performance metrics
    recovery_analyzer.print_performance_metrics()
    
    # Print progressive recovery summary
    recovery_analyzer.print_progressive_recovery_summary()
    
    # Print adaptive timeouts
    recovery_analyzer.print_adaptive_timeouts()


if __name__ == "__main__":
    anyio.run(main())