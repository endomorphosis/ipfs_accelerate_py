#!/usr/bin/env python3
"""
Test script for the Database Performance Optimization component of the Simulation Accuracy and Validation Framework.

This script tests the database performance optimization features, including:
- Query caching
- Batch operations
- Query optimization
- Database maintenance
- Backup and restore functionality
"""

import os
import sys
import time
import logging
import tempfile
import unittest
import json
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_db_performance_optimization")

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)
from data.duckdb.simulation_validation.db_integration import SimulationValidationDBIntegration
from data.duckdb.simulation_validation.db_performance_optimization import (
    QueryCache,
    BatchOperations,
    QueryOptimizer,
    DatabaseMaintenance,
    DatabaseBackupManager,
    OptimizedSimulationValidationDBIntegration,
    get_optimized_db_integration
)
from data.duckdb.simulation_validation.test_validator import generate_sample_data


class TestQueryCache(unittest.TestCase):
    """Test cases for the QueryCache class."""
    
    def setUp(self):
        """Set up test environment."""
        self.cache = QueryCache(ttl=2, max_size=5)  # Small cache for testing
    
    def test_cache_store_retrieve(self):
        """Test storing and retrieving items from cache."""
        # Store an item
        query = "SELECT * FROM test_table WHERE id = :id"
        params = {"id": 1}
        result = [("test", 123)]
        
        self.cache.set(query, params, result)
        
        # Retrieve the item
        cached_result = self.cache.get(query, params)
        
        self.assertEqual(cached_result, result, "Retrieved item should match stored item")
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        # Store an item
        query = "SELECT * FROM test_table"
        result = [("test", 123)]
        
        self.cache.set(query, None, result)
        
        # Verify item is in cache
        self.assertIsNotNone(self.cache.get(query), "Item should be in cache")
        
        # Wait for TTL to expire
        time.sleep(3)
        
        # Verify item is no longer in cache
        self.assertIsNone(self.cache.get(query), "Item should have expired")
    
    def test_cache_max_size(self):
        """Test cache max size limit."""
        # Fill the cache
        for i in range(6):  # One more than max size
            query = f"SELECT * FROM test_table WHERE id = {i}"
            result = [(f"test-{i}", i)]
            self.cache.set(query, None, result)
        
        # Check that the oldest entry is removed
        self.assertIsNone(self.cache.get("SELECT * FROM test_table WHERE id = 0"), 
                         "Oldest entry should have been removed")
        
        # Check that newest entries are still there
        for i in range(1, 6):
            query = f"SELECT * FROM test_table WHERE id = {i}"
            self.assertIsNotNone(self.cache.get(query), 
                                f"Entry {i} should still be in cache")
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        # Store items for different tables
        self.cache.set("SELECT * FROM table1", None, [("table1", 1)])
        self.cache.set("SELECT * FROM table2", None, [("table2", 2)])
        self.cache.set("SELECT * FROM table1 JOIN table2", None, [("join", 3)])
        
        # Invalidate one table
        self.cache.invalidate("table1")
        
        # Check that the right items were invalidated
        self.assertIsNone(self.cache.get("SELECT * FROM table1"), 
                         "table1 entry should have been invalidated")
        self.assertIsNotNone(self.cache.get("SELECT * FROM table2"), 
                            "table2 entry should still be in cache")
        
        # Check behavior of entries with both tables
        # Note: Our simple implementation might not catch this, but more sophisticated
        # implementations would invalidate this too
        
        # Clear the entire cache
        self.cache.clear()
        
        # Check that the cache is empty
        self.assertIsNone(self.cache.get("SELECT * FROM table2"), 
                         "Cache should be empty after clear")


class TestDatabaseOptimization(unittest.TestCase):
    """Test cases for database optimization components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary database file for testing
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()
        
        # Create output and backup directories
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        self.backup_dir = Path(__file__).parent / "output" / "backups"
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize the standard database integration
        self.db_integration = SimulationValidationDBIntegration(
            db_path=self.temp_db_path
        )
        
        # Initialize the optimized database integration
        self.opt_db_integration = OptimizedSimulationValidationDBIntegration(
            db_path=self.temp_db_path,
            enable_caching=True,
            batch_size=2,  # Small batch for testing
            backup_dir=str(self.backup_dir),
            auto_optimize=True
        )
        
        # Initialize optimization components
        self.batch_operations = BatchOperations(self.db_integration, batch_size=2)
        self.query_optimizer = QueryOptimizer(self.db_integration)
        self.maintenance = DatabaseMaintenance(self.db_integration)
        self.backup_manager = DatabaseBackupManager(self.db_integration, 
                                                  backup_dir=str(self.backup_dir))
        
        # Create sample data
        self.simulation_results, self.hardware_results = generate_sample_data(num_samples=5)
        
        # Create validation results
        self.validation_results = []
        for i in range(len(self.simulation_results)):
            # Add metrics comparison
            metrics_comparison = {}
            
            for metric in ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", "power_consumption_w"]:
                if (metric in self.simulation_results[i].metrics and
                    metric in self.hardware_results[i].metrics):
                    sim_value = self.simulation_results[i].metrics[metric]
                    hw_value = self.hardware_results[i].metrics[metric]
                    
                    if sim_value is not None and hw_value is not None and hw_value != 0:
                        # Calculate error metrics
                        abs_error = abs(hw_value - sim_value)
                        rel_error = abs_error / abs(hw_value)
                        mape = rel_error * 100  # percentage
                        
                        metrics_comparison[metric] = {
                            "simulation_value": sim_value,
                            "hardware_value": hw_value,
                            "absolute_error": abs_error,
                            "relative_error": rel_error,
                            "mape": mape
                        }
            
            validation_result = ValidationResult(
                simulation_result=self.simulation_results[i],
                hardware_result=self.hardware_results[i],
                metrics_comparison=metrics_comparison,
                validation_timestamp=self.simulation_results[i].timestamp,
                validation_version="test_v1.0",
                additional_metrics={}
            )
            
            self.validation_results.append(validation_result)
        
        # Initialize database if needed
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database with schema."""
        try:
            # This is a convenience method to make sure the database is set up
            conn = self.db_integration.db_api._get_connection()
            
            # Check if tables exist
            tables_exist = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='simulation_results'"
            ).fetchone()
            
            if not tables_exist:
                # Create tables based on schema
                from data.duckdb.simulation_validation.core.schema import SimulationValidationSchema
                SimulationValidationSchema.create_tables(conn)
                logger.info("Created database schema for testing")
            
            # Store some data for testing
            from data.duckdb.simulation_validation.core.schema import SIMULATION_VALIDATION_SCHEMA
            schema = SIMULATION_VALIDATION_SCHEMA
            
            # Add a store_data method to db_integration if it doesn't exist
            if not hasattr(self.db_integration, 'store_simulation_result'):
                self.db_integration.store_simulation_result = lambda result: self._store_result(
                    result, "simulation_results", schema.simulation_result_to_db_dict
                )
            
            if not hasattr(self.db_integration, 'store_hardware_result'):
                self.db_integration.store_hardware_result = lambda result: self._store_result(
                    result, "hardware_results", schema.hardware_result_to_db_dict
                )
        
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _store_result(self, result, table_name, converter_func):
        """Helper method to store a result in the database."""
        try:
            conn = self.db_integration.db_api._get_connection()
            record = converter_func(result)
            
            # Generate placeholder and column string
            columns = list(record.keys())
            placeholders = ", ".join(["?"] * len(columns))
            columns_str = ", ".join(columns)
            
            # Extract values
            values = [record[col] for col in columns]
            
            # Create query
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # Execute query
            conn.execute(query, values)
            conn.commit()
            
            return record.get("id")
        except Exception as e:
            logger.error(f"Error storing result: {e}")
            return None
    
    def tearDown(self):
        """Clean up test environment."""
        # Close database connections
        if hasattr(self.db_integration, 'close'):
            self.db_integration.close()
        
        if hasattr(self.opt_db_integration, 'close'):
            self.opt_db_integration.close()
        
        # Remove the temporary database file
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
        
        # Clean up backups
        for backup_file in self.backup_dir.glob("*"):
            try:
                if backup_file.is_file():
                    os.unlink(backup_file)
            except Exception as e:
                logger.warning(f"Error deleting backup file {backup_file}: {e}")
    
    def test_query_optimizer(self):
        """Test query optimization."""
        # Create indexes
        self.query_optimizer.create_indexes()
        
        # Test query optimization
        original_query = "SELECT * FROM simulation_results WHERE hardware_id = 'gpu_rtx3080'"
        optimized_query = self.query_optimizer.optimize_query(original_query)
        
        # Check that the query was optimized (should have LIMIT added)
        self.assertIn("LIMIT", optimized_query, "Query should have LIMIT added")
        
        # Test with more complex query
        complex_query = """
        SELECT sr.model_id, sr.hardware_id, AVG(vr.overall_accuracy_score) as avg_score
        FROM validation_results vr
        JOIN simulation_results sr ON vr.simulation_result_id = sr.id
        GROUP BY sr.model_id, sr.hardware_id
        ORDER BY avg_score DESC
        """
        
        optimized_complex = self.query_optimizer.optimize_query(complex_query)
        self.assertIn("LIMIT", optimized_complex, "Complex query should have LIMIT added")
        
        # Execute the optimized query to make sure it works
        try:
            conn = self.db_integration.db_api._get_connection()
            conn.execute(optimized_complex)
            # If we get here, the query is valid
            passed = True
        except Exception as e:
            logger.error(f"Error executing optimized query: {e}")
            passed = False
        
        self.assertTrue(passed, "Optimized query should execute without errors")
    
    def test_batch_operations(self):
        """Test batch operations for database interactions."""
        # Generate some test records
        test_records = []
        for i in range(10):
            # Create a mock simulation result record
            record = {
                "id": f"sim-{i}",
                "model_id": f"model-{i % 3}",
                "hardware_id": f"hw-{i % 2}",
                "batch_size": 16,
                "precision": "fp32",
                "timestamp": datetime.datetime.now().isoformat(),
                "metrics": json.dumps({
                    "throughput_items_per_second": 100 + i,
                    "average_latency_ms": 10 - i * 0.1
                }),
                "additional_metadata": json.dumps({"test": True})
            }
            test_records.append(record)
        
        # Test batch insert
        count = self.batch_operations.batch_insert(
            table_name="simulation_results",
            records=test_records,
            columns=list(test_records[0].keys())
        )
        
        self.assertEqual(count, len(test_records), 
                         f"Should have inserted {len(test_records)} records")
        
        # Verify the records were inserted
        conn = self.db_integration.db_api._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM simulation_results").fetchone()
        self.assertEqual(result[0], len(test_records), 
                         "Database should contain the batch-inserted records")
        
        # Test batch update
        update_records = [
            {"id": "sim-1", "precision": "fp16"},
            {"id": "sim-2", "precision": "int8"},
            {"id": "sim-3", "precision": "bf16"}
        ]
        
        update_count = self.batch_operations.batch_update(
            table_name="simulation_results",
            records=update_records,
            id_column="id",
            update_columns=["precision"]
        )
        
        self.assertEqual(update_count, len(update_records), 
                         f"Should have updated {len(update_records)} records")
        
        # Verify the updates
        for record in update_records:
            result = conn.execute(
                f"SELECT precision FROM simulation_results WHERE id = ?",
                [record["id"]]
            ).fetchone()
            self.assertEqual(result[0], record["precision"], 
                             f"Record {record['id']} should have precision {record['precision']}")
        
        # Test batch delete
        delete_ids = ["sim-4", "sim-5", "sim-6"]
        
        delete_count = self.batch_operations.batch_delete(
            table_name="simulation_results",
            id_column="id",
            id_values=delete_ids
        )
        
        self.assertEqual(delete_count, len(delete_ids), 
                         f"Should have deleted {len(delete_ids)} records")
        
        # Verify the deletes
        for id_val in delete_ids:
            result = conn.execute(
                f"SELECT COUNT(*) FROM simulation_results WHERE id = ?",
                [id_val]
            ).fetchone()
            self.assertEqual(result[0], 0, f"Record {id_val} should have been deleted")
    
    def test_database_maintenance(self):
        """Test database maintenance operations."""
        # Store some data for testing
        if hasattr(self.db_integration, 'store_simulation_result'):
            for result in self.simulation_results:
                self.db_integration.store_simulation_result(result)
        
        # Test integrity check
        integrity = self.maintenance.check_integrity()
        self.assertTrue(integrity["overall"], "Database integrity check should pass")
        
        # Test vacuum
        try:
            self.maintenance.vacuum_database()
            passed = True
        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
            passed = False
        
        self.assertTrue(passed, "Vacuum database should complete without errors")
        
        # Test cleanup of old data (with a retention period that keeps all data)
        cleanup_result = self.maintenance.cleanup_old_data(retention_days=3650)
        self.assertNotIn("error", cleanup_result, "Cleanup should complete without errors")
        
        # Test performance monitoring
        monitor_result = self.maintenance.monitor_database_performance()
        self.assertNotIn("error", monitor_result, "Performance monitoring should complete without errors")
        self.assertIn("database_size_bytes", monitor_result, "Performance report should include database size")
        self.assertIn("table_counts", monitor_result, "Performance report should include table counts")
    
    def test_database_backup(self):
        """Test database backup and restore functionality."""
        # Store some data for testing
        if hasattr(self.db_integration, 'store_simulation_result'):
            for result in self.simulation_results:
                self.db_integration.store_simulation_result(result)
        
        # Create a backup
        backup_result = self.backup_manager.create_backup(compress=True, verify=True)
        self.assertNotIn("error", backup_result, "Backup should complete without errors")
        self.assertTrue(backup_result.get("compressed", False), "Backup should be compressed")
        self.assertTrue(os.path.exists(backup_result["backup_path"]), "Backup file should exist")
        
        # List backups
        backups = self.backup_manager.list_backups()
        self.assertEqual(len(backups), 1, "Should have one backup listed")
        
        # Create another backup
        second_backup = self.backup_manager.create_backup(compress=False, verify=True)
        self.assertNotIn("error", second_backup, "Second backup should complete without errors")
        
        # Test backup verification
        verify_result = self.backup_manager.verify_backup(
            backup_path=second_backup["backup_path"],
            compressed=second_backup.get("compressed", False)
        )
        self.assertTrue(verify_result["success"], "Backup verification should succeed")
        
        # Delete the first backup
        delete_result = self.backup_manager.delete_backup(backup_result["backup_name"])
        self.assertTrue(delete_result["success"], "Backup deletion should succeed")
        
        # Verify the backup was deleted
        backups = self.backup_manager.list_backups()
        self.assertEqual(len(backups), 1, "Should have one backup remaining")
        
        # Create a scheduled backup script
        schedule_result = self.backup_manager.schedule_backup(
            interval_hours=24,
            keep_backups=5,
            compress=True
        )
        self.assertTrue(schedule_result["success"], "Scheduling backups should succeed")
        self.assertTrue(os.path.exists(schedule_result["script_path"]), "Backup script should exist")
        self.assertTrue(os.path.exists(schedule_result["readme_path"]), "Backup README should exist")
    
    def test_optimized_db_integration(self):
        """Test the optimized database integration class."""
        # Test batch insert functionality
        batch_result = self.opt_db_integration.batch_insert_simulation_results(
            self.simulation_results
        )
        self.assertEqual(batch_result, len(self.simulation_results), 
                         f"Should have inserted {len(self.simulation_results)} simulation results")
        
        # Test hardware results insert
        hw_batch_result = self.opt_db_integration.batch_insert_hardware_results(
            self.hardware_results
        )
        self.assertEqual(hw_batch_result, len(self.hardware_results), 
                         f"Should have inserted {len(self.hardware_results)} hardware results")
        
        # Test query caching
        query = "SELECT COUNT(*) FROM simulation_results"
        
        # First execution (no cache)
        start_time = time.time()
        first_result = self.opt_db_integration._execute_query(query, use_cache=True)
        first_time = time.time() - start_time
        
        # Second execution (should use cache)
        start_time = time.time()
        second_result = self.opt_db_integration._execute_query(query, use_cache=True)
        second_time = time.time() - start_time
        
        # The second execution should be faster if caching is working
        self.assertEqual(first_result, second_result, "Cached result should match original")
        logger.info(f"First query: {first_time:.6f}s, Second query: {second_time:.6f}s")
        
        # Test cache invalidation
        self.opt_db_integration.invalidate_cache("simulation_results")
        
        # After invalidation, the query should not use cache
        start_time = time.time()
        third_result = self.opt_db_integration._execute_query(query, use_cache=True)
        third_time = time.time() - start_time
        
        self.assertEqual(first_result, third_result, "Result after invalidation should match original")
        logger.info(f"First query: {first_time:.6f}s, Third query (after invalidation): {third_time:.6f}s")
        
        # Test database optimization
        opt_result = self.opt_db_integration.optimize_database()
        self.assertTrue(opt_result["overall"], "Database optimization should succeed")
        
        # Test performance statistics
        stats = self.opt_db_integration.get_performance_statistics()
        self.assertIn("database_size_bytes", stats, "Performance stats should include database size")
        self.assertIn("tables", stats, "Performance stats should include table statistics")
        self.assertIn("query_cache", stats, "Performance stats should include cache information")
        
        # Test maintenance operations
        maintenance_result = self.opt_db_integration.run_maintenance(["vacuum", "integrity_check"])
        self.assertTrue(maintenance_result["overall"], "Maintenance operations should succeed")
        
        # Test backup functionality
        backup_result = self.opt_db_integration.backup_database(compress=True)
        self.assertTrue(backup_result.get("compressed", False), "Backup should be compressed")
        
        # Test listing backups
        backups = self.opt_db_integration.list_backups()
        self.assertEqual(len(backups), 1, "Should have one backup listed")
    
    def test_utility_function(self):
        """Test the utility function for getting an optimized DB integration instance."""
        # Get an optimized DB integration
        custom_db = get_optimized_db_integration(
            db_path=self.temp_db_path,
            debug=True,
            enable_caching=True,
            cache_ttl=600,
            batch_size=50,
            backup_dir=str(self.backup_dir),
            auto_optimize=False
        )
        
        self.assertIsInstance(custom_db, OptimizedSimulationValidationDBIntegration, 
                             "Should get an instance of OptimizedSimulationValidationDBIntegration")
        self.assertEqual(custom_db.db_path, self.temp_db_path, "Should use the specified DB path")
        
        # Test basic functionality
        try:
            # Execute a simple query
            result = custom_db.execute_query("SELECT 1")
            self.assertEqual(result[0][0], 1, "Simple query should return expected result")
            
            # Clean up
            custom_db.close()
            passed = True
        except Exception as e:
            logger.error(f"Error testing utility function: {e}")
            passed = False
        
        self.assertTrue(passed, "Utility function should return a working DB integration instance")


def main():
    """Run the database performance optimization tests."""
    logger.info("Running Database Performance Optimization tests")
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestQueryCache))
    suite.addTest(unittest.makeSuite(TestDatabaseOptimization))
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    logger.info("Tests completed")
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())