#!/usr/bin/env python3
"""
Test Database Integration for Error Recovery System

This script tests the database integration for the performance-based error
recovery system, ensuring that performance metrics are properly stored
and retrieved from the database.
"""

import os
import sys
import logging
import anyio
import time
import json
import uuid
from datetime import datetime, timedelta
import random
import pytest

pytestmark = pytest.mark.anyio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_error_recovery_db")

# Import required modules
try:
    import duckdb
    try:
        # Prefer package imports when collected by pytest
        from test.tests.distributed.distributed_testing.distributed_error_handler import (
            DistributedErrorHandler,
            ErrorType,
            ErrorSeverity,
        )
        from test.tests.distributed.distributed_testing.error_recovery_strategies import EnhancedErrorRecoveryManager
        from test.tests.distributed.distributed_testing.error_recovery_with_performance_tracking import (
            PerformanceBasedErrorRecovery,
        )
        from test.tests.distributed.distributed_testing.enhanced_error_handling_integration import (
            install_enhanced_error_handling,
        )
    except ImportError:
        # Fallback for running this file directly from this directory
        from distributed_error_handler import DistributedErrorHandler, ErrorType, ErrorSeverity
        from error_recovery_strategies import EnhancedErrorRecoveryManager
        from error_recovery_with_performance_tracking import PerformanceBasedErrorRecovery
        from enhanced_error_handling_integration import install_enhanced_error_handling
except ImportError as e:
    pytest.skip(
        f"Skipping distributed error recovery DB integration tests (missing optional deps): {e}. "
        "Tip: install test dependencies with `pip install -e '.[testing]'`.",
        allow_module_level=True,
    )


class MockCoordinator:
    """Mock coordinator for testing."""
    
    def __init__(self, db_path=":memory:"):
        """Initialize mock coordinator."""
        self.tasks = {}
        self.running_tasks = {}
        self.pending_tasks = set()
        self.worker_connections = {}
        self.workers = {}
        
        # Initialize database
        self.db = duckdb.connect(db_path)
        
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


async def test_database_schema_creation():
    """Test database schema creation and migration."""
    logger.info("Testing database schema creation")
    
    # Create coordinator with in-memory database
    coordinator = MockCoordinator()
    
    # Create error recovery components
    error_handler = DistributedErrorHandler()
    recovery_manager = EnhancedErrorRecoveryManager(coordinator)
    
    # Create performance recovery system
    recovery = PerformanceBasedErrorRecovery(
        error_handler=error_handler,
        recovery_manager=recovery_manager,
        coordinator=coordinator,
        db_connection=coordinator.db
    )
    
    # Check that tables were created
    tables = coordinator.db.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table'
    """).fetchall()
    
    tables = [t[0] for t in tables]
    
    required_tables = [
        "recovery_performance",
        "strategy_scores",
        "adaptive_timeouts",
        "progressive_recovery",
        "schema_versions"
    ]
    
    # Check that all required tables exist
    for table in required_tables:
        if table not in tables:
            logger.error(f"Table {table} not found in database")
            return False
    
    # Check schema version
    version = coordinator.db.execute("""
    SELECT version FROM schema_versions 
    WHERE component='performance_recovery'
    """).fetchone()
    
    if not version or version[0] != 1:
        logger.error(f"Incorrect schema version: {version}")
        return False
    
    logger.info("Database schema creation successful")
    return True


async def test_performance_metrics_storage():
    """Test storage and retrieval of performance metrics."""
    logger.info("Testing performance metrics storage")
    
    # Create coordinator with in-memory database
    coordinator = MockCoordinator()
    
    # Install enhanced error handling
    error_handling = install_enhanced_error_handling(coordinator)
    
    # Generate sample errors and performance data
    error_types = ["network", "database", "timeout", "resource", "coordinator"]
    success_rates = {"network": 0.7, "database": 0.8, "timeout": 0.9, "resource": 0.6, "coordinator": 0.5}
    
    # Store performance data
    for _ in range(20):
        error_type = random.choice(error_types)
        success = random.random() < success_rates[error_type]
        
        # Create mock error
        error = Exception(f"Test {error_type} error")
        
        # Handle error
        await error_handling.handle_error(error, {
            "component": error_type,
            "operation": "test_operation",
            "error_type": error_type
        })
    
    # Query performance metrics
    metrics = error_handling.get_performance_metrics()
    
    # Check that metrics were stored
    if not metrics["strategies"]:
        logger.error("No performance metrics stored")
        return False
    
    # Check that metrics can be retrieved by error type
    for error_type in error_types:
        type_metrics = error_handling.get_performance_metrics(error_type=error_type)
        
        # Some error types might not have been used
        if type_metrics["strategies"]:
            logger.info(f"Found metrics for {error_type}")
    
    logger.info("Performance metrics storage successful")
    return True


async def test_schema_migration():
    """Test schema migration."""
    logger.info("Testing schema migration")
    
    # Create temporary database file
    db_path = "test_migration.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # Create coordinator with file database
        coordinator = MockCoordinator(db_path)
        
        # Create outdated version of the table
        coordinator.db.execute("""
        CREATE TABLE recovery_performance (
            id INTEGER PRIMARY KEY,
            strategy_id VARCHAR,
            strategy_name VARCHAR,
            error_type VARCHAR,
            execution_time FLOAT,
            success BOOLEAN,
            timestamp TIMESTAMP
        )
        """)
        
        # Insert some data
        coordinator.db.execute("""
        INSERT INTO recovery_performance (id, strategy_id, strategy_name, error_type, execution_time, success, timestamp)
        VALUES (1, 'retry', 'retry', 'network', 0.5, 1, CURRENT_TIMESTAMP)
        """)
        
        # Create schema versions table for tracking
        coordinator.db.execute("""
        CREATE TABLE schema_versions (
            component VARCHAR PRIMARY KEY, 
            version INTEGER,
            last_updated TIMESTAMP
        )
        """)
        
        # Insert schema version
        coordinator.db.execute("""
        INSERT INTO schema_versions (component, version, last_updated)
        VALUES ('performance_recovery', 0, CURRENT_TIMESTAMP)
        """)
        
        # Create recovery system (should trigger migration)
        error_handler = DistributedErrorHandler()
        recovery_manager = EnhancedErrorRecoveryManager(coordinator)
        
        recovery = PerformanceBasedErrorRecovery(
            error_handler=error_handler,
            recovery_manager=recovery_manager,
            coordinator=coordinator,
            db_connection=coordinator.db
        )
        
        # Check that migration was successful
        count = coordinator.db.execute("""
        SELECT COUNT(*) FROM recovery_performance
        """).fetchone()[0]
        
        if count != 1:
            logger.error(f"Data was lost during migration: found {count} rows, expected 1")
            return False
        
        # Check schema version
        version = coordinator.db.execute("""
        SELECT version FROM schema_versions 
        WHERE component='performance_recovery'
        """).fetchone()
        
        if not version or version[0] != 1:
            logger.error(f"Incorrect schema version after migration: {version}")
            return False
        
        # Try to insert a new record (should provide a manually generated ID since DuckDB doesn't support AUTOINCREMENT)
        coordinator.db.execute("""
        INSERT INTO recovery_performance 
        (id, strategy_id, strategy_name, error_type, execution_time, success, timestamp)
        VALUES (2, 'retry', 'retry', 'network', 0.5, 1, CURRENT_TIMESTAMP)
        """)
        
        # Check that the new record was inserted
        count = coordinator.db.execute("""
        SELECT COUNT(*) FROM recovery_performance
        """).fetchone()[0]
        
        if count != 2:
            logger.error(f"Failed to insert new record after migration: found {count} rows, expected 2")
            return False
        
        logger.info("Schema migration successful")
        return True
    
    finally:
        # Clean up temporary database file
        if os.path.exists(db_path):
            os.remove(db_path)


async def test_date_functions():
    """Test DuckDB date functions."""
    logger.info("Testing DuckDB date functions")
    
    # Create coordinator with in-memory database
    coordinator = MockCoordinator()
    
    try:
        # Test date interval function
        result = coordinator.db.execute("""
        SELECT (CURRENT_TIMESTAMP - INTERVAL '30 days') as thirty_days_ago
        """).fetchone()
        
        thirty_days_ago = result[0]
        now = datetime.now()
        
        # Check that the date is approximately 30 days ago
        # Convert to datetime.datetime objects if needed
        if hasattr(thirty_days_ago, 'replace'):
            thirty_days_ago = thirty_days_ago.replace(tzinfo=None)
        
        delta = now - thirty_days_ago
        if delta.days < 29 or delta.days > 31:
            logger.error(f"Date function returned unexpected result: {thirty_days_ago}, delta: {delta.days} days")
            return False
        
        # Create a table with timestamps
        coordinator.db.execute("""
        CREATE TABLE date_test (
            id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP
        )
        """)
        
        # Insert some data
        for days_ago in [1, 5, 10, 20, 30, 40, 50]:
            query = f"""
            INSERT INTO date_test (id, timestamp)
            VALUES ({days_ago}, CURRENT_TIMESTAMP - INTERVAL '{days_ago} days')
            """
            coordinator.db.execute(query)
        
        # Test query with date filter
        for days in [15, 30, 45]:
            query = f"""
            SELECT COUNT(*) FROM date_test
            WHERE timestamp > (CURRENT_TIMESTAMP - INTERVAL '{days} days')
            """
            count = coordinator.db.execute(query).fetchone()[0]
            
            expected = sum(1 for d in [1, 5, 10, 20, 30, 40, 50] if d < days)
            
            if count != expected:
                logger.error(f"Date filter returned {count} rows, expected {expected}")
                return False
        
        logger.info("DuckDB date functions test successful")
        return True
    
    except Exception as e:
        logger.error(f"Error testing date functions: {str(e)}")
        return False


async def test_recovery_history_persistence():
    """Test persistence of recovery history."""
    logger.info("Testing recovery history persistence")
    
    # Create temporary database file
    db_path = "test_recovery.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # Create coordinator with file database
        coordinator = MockCoordinator(db_path)
        
        # Install enhanced error handling
        error_handling = install_enhanced_error_handling(coordinator)
        
        # Create mock error to persist with progressive recovery
        error = Exception("Test persistent error")
        error_id = None
        
        # Handle error first time (will fail)
        for i in range(3):
            success, info = await error_handling.handle_error(error, {
                "component": "network",
                "operation": "connect",
                "error_type": "network"
            })
            
            if i == 0:
                # Save error ID for later
                error_id = info.get("error_id")
            
            if success:
                break
                
            # Wait a bit between retries
            await anyio.sleep(0.1)
        
        # Check recovery history
        if not error_id:
            logger.error("No error ID returned")
            return False
        
        history = error_handling.get_recovery_history(error_id)
        
        if not history or "history" not in history:
            logger.error("No recovery history found")
            return False
        
        # Close database
        coordinator.db.close()
        
        # Reopen database with new coordinator
        coordinator2 = MockCoordinator(db_path)
        
        # Install enhanced error handling
        error_handling2 = install_enhanced_error_handling(coordinator2)
        
        # Check that history is still available
        history2 = error_handling2.get_recovery_history(error_id)
        
        if not history2 or "history" not in history2:
            logger.error("Recovery history not persisted")
            return False
        
        if len(history2["history"]) != len(history["history"]):
            logger.error(f"History length mismatch: {len(history2['history'])} vs {len(history['history'])}")
            return False
        
        logger.info("Recovery history persistence successful")
        return True
    
    finally:
        # Clean up temporary database file
        if os.path.exists(db_path):
            os.remove(db_path)


async def test_performance_record_integrity():
    """Test integrity of performance records."""
    logger.info("Testing performance record integrity")
    
    # Create coordinator with in-memory database
    coordinator = MockCoordinator()
    
    # Install enhanced error handling
    error_handling = install_enhanced_error_handling(coordinator)
    
    # Generate sample errors
    for _ in range(10):
        error_type = random.choice(["network", "database", "timeout"])
        error = Exception(f"Test {error_type} error")
        
        # Handle error
        await error_handling.handle_error(error, {
            "component": error_type,
            "operation": "test_operation",
            "error_type": error_type
        })
    
    # Count records
    count = coordinator.db.execute("""
    SELECT COUNT(*) FROM recovery_performance
    """).fetchone()[0]
    
    if count < 10:
        logger.error(f"Expected at least 10 records, found {count}")
        return False
    
    # Check for NULL values in required fields
    nulls = coordinator.db.execute("""
    SELECT COUNT(*) FROM recovery_performance
    WHERE strategy_id IS NULL OR strategy_name IS NULL OR error_type IS NULL
    """).fetchone()[0]
    
    if nulls > 0:
        logger.error(f"Found {nulls} records with NULL values in required fields")
        return False
    
    # Verify JSON fields
    try:
        records = coordinator.db.execute("""
        SELECT resource_usage, context FROM recovery_performance
        """).fetchall()
        
        for record in records:
            resource_usage, context = record
            
            # Parse JSON
            if resource_usage:
                json.loads(resource_usage)
            
            if context:
                json.loads(context)
    
    except Exception as e:
        logger.error(f"Error parsing JSON fields: {str(e)}")
        return False
    
    logger.info("Performance record integrity test successful")
    return True


async def run_all_tests():
    """Run all database integration tests."""
    logger.info("Running all database integration tests")
    
    # Test database schema creation
    schema_result = await test_database_schema_creation()
    logger.info(f"Schema creation test: {'SUCCESS' if schema_result else 'FAILURE'}")
    
    # Test date functions
    date_result = await test_date_functions()
    logger.info(f"Date functions test: {'SUCCESS' if date_result else 'FAILURE'}")
    
    # Test schema migration
    migration_result = await test_schema_migration()
    logger.info(f"Schema migration test: {'SUCCESS' if migration_result else 'FAILURE'}")
    
    # Test performance metrics storage
    metrics_result = await test_performance_metrics_storage()
    logger.info(f"Performance metrics storage test: {'SUCCESS' if metrics_result else 'FAILURE'}")
    
    # Test recovery history persistence
    history_result = await test_recovery_history_persistence()
    logger.info(f"Recovery history persistence test: {'SUCCESS' if history_result else 'FAILURE'}")
    
    # Test performance record integrity
    integrity_result = await test_performance_record_integrity()
    logger.info(f"Performance record integrity test: {'SUCCESS' if integrity_result else 'FAILURE'}")
    
    # Overall result
    all_passed = all([
        schema_result,
        date_result,
        migration_result,
        metrics_result,
        history_result,
        integrity_result
    ])
    
    print("\n" + "="*50)
    print(f"OVERALL RESULT: {'SUCCESS' if all_passed else 'FAILURE'}")
    print("="*50)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test database integration for error recovery system")
    parser.add_argument("--test", choices=["schema", "date", "migration", "metrics", "history", "integrity", "all"], 
                        default="all", help="Specific test to run")
    args = parser.parse_args()
    
    if args.test == "schema":
        result = anyio.run(test_database_schema_creation())
    elif args.test == "date":
        result = anyio.run(test_date_functions())
    elif args.test == "migration":
        result = anyio.run(test_schema_migration())
    elif args.test == "metrics":
        result = anyio.run(test_performance_metrics_storage())
    elif args.test == "history":
        result = anyio.run(test_recovery_history_persistence())
    elif args.test == "integrity":
        result = anyio.run(test_performance_record_integrity())
    else:
        result = anyio.run(run_all_tests())
    
    sys.exit(0 if result else 1)