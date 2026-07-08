#!/usr/bin/env python3
"""
Database Performance Optimization Module for the Simulation Accuracy and Validation Framework.

This module provides tools and utilities for optimizing the performance of the DuckDB database
backend used by the Simulation Accuracy and Validation Framework. It includes:

1. Query optimization for large datasets
2. Batch operations for improved efficiency
3. Query caching for frequently accessed data
4. Database maintenance utilities
5. Database backup and restore functionality
"""

import os
import sys
import time
import json
import logging
import datetime
import hashlib
import shutil
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_performance_optimizer")

# Import the database integration module
try:
    from data.duckdb.simulation_validation.db_integration import SimulationValidationDBIntegration
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
except ImportError:
    logger.error("Failed to import SimulationValidationDBIntegration. Make sure duckdb_api is properly installed.")
    sys.exit(1)

# Import base classes for type checking
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)


class QueryCache:
    """
    Implements a caching mechanism for database queries to improve performance.
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        """
        Initialize the query cache.
        
        Args:
            max_size: Maximum number of cached results to store
            ttl: Time-to-live in seconds for cached results
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.RLock()
    
    def get(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get a result from the cache.
        
        Args:
            query: The SQL query string
            params: The query parameters
            
        Returns:
            The cached result or None if not found
        """
        cache_key = self._generate_cache_key(query, params)
        
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                # Check if the entry is still valid
                if time.time() - entry["timestamp"] <= self.ttl:
                    self.cache_hits += 1
                    return entry["result"]
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def set(self, query: str, params: Dict[str, Any], result: Any) -> None:
        """
        Store a result in the cache.
        
        Args:
            query: The SQL query string
            params: The query parameters
            result: The result to cache
        """
        cache_key = self._generate_cache_key(query, params)
        
        with self.lock:
            # Ensure we don't exceed max_size
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.items(), key=lambda x: x[1]["timestamp"])[0]
                del self.cache[oldest_key]
            
            # Store the new entry
            self.cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
    
    def invalidate(self, table_name: Optional[str] = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            table_name: If provided, only invalidate entries for this table
        """
        with self.lock:
            if table_name is None:
                # Invalidate all entries
                self.cache.clear()
                logger.info("Cleared entire query cache")
            else:
                # Invalidate entries for the specified table
                keys_to_remove = []
                for key in self.cache:
                    if table_name.lower() in key.lower():
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.cache[key]
                
                logger.info(f"Cleared {len(keys_to_remove)} entries from query cache for table {table_name}")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """
        Generate a unique key for the query and parameters.
        
        Args:
            query: The SQL query string
            params: The query parameters
            
        Returns:
            A unique cache key
        """
        # Normalize query (remove whitespace)
        normalized_query = " ".join(query.split())
        
        # Convert params to a string
        params_str = json.dumps(params, sort_keys=True)
        
        # Generate hash
        key_str = f"{normalized_query}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()


class BatchOperation:
    """
    Implements batch operations for improved efficiency when inserting/updating
    multiple database records.
    """
    
    def __init__(self, db_api: BenchmarkDBAPI, batch_size: int = 100):
        """
        Initialize the batch operation handler.
        
        Args:
            db_api: BenchmarkDBAPI instance
            batch_size: Number of operations to batch together
        """
        self.db_api = db_api
        self.batch_size = batch_size
        self.operations = []
        self.pending_count = 0
    
    def add_operation(self, query: str, params: Dict[str, Any]) -> None:
        """
        Add an operation to the batch.
        
        Args:
            query: SQL query to execute
            params: Query parameters
        """
        self.operations.append((query, params))
        self.pending_count += 1
        
        # Execute if we've reached the batch size
        if self.pending_count >= self.batch_size:
            self.execute()
    
    def execute(self) -> None:
        """
        Execute all pending operations as a single transaction.
        """
        if not self.operations:
            return
        
        try:
            conn = self.db_api._get_connection()
            
            # Start a transaction
            with conn:
                for query, params in self.operations:
                    conn.execute(query, params)
            
            logger.info(f"Successfully executed batch of {len(self.operations)} operations")
            
            # Clear the operations
            operations_count = len(self.operations)
            self.operations = []
            self.pending_count = 0
            
            return operations_count
        except Exception as e:
            logger.error(f"Error executing batch operations: {e}")
            # Clear operations to avoid retrying the same failed batch
            self.operations = []
            self.pending_count = 0
            raise


class DBPerformanceOptimizer:
    """
    Main class for optimizing database performance.
    """
    
    def __init__(
        self, 
        db_path: str = "./benchmark_db.duckdb", 
        enable_caching: bool = True,
        cache_size: int = 100,
        cache_ttl: int = 300,
        batch_size: int = 100
    ):
        """
        Initialize the database performance optimizer.
        
        Args:
            db_path: Path to the DuckDB database file
            enable_caching: Whether to enable query caching
            cache_size: Maximum number of cached results to store
            cache_ttl: Time-to-live in seconds for cached results
            batch_size: Default batch size for batch operations
        """
        self.db_path = db_path
        self.enable_caching = enable_caching
        
        # Initialize database integration
        try:
            self.db_integration = SimulationValidationDBIntegration(db_path=db_path)
            logger.info(f"Connected to database at {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.db_integration = None
        
        # Initialize query cache
        self.cache = QueryCache(max_size=cache_size, ttl=cache_ttl) if enable_caching else None
        
        # Initialize batch operations handler
        if self.db_integration and hasattr(self.db_integration, 'db_api'):
            self.batch_handler = BatchOperation(self.db_integration.db_api, batch_size=batch_size)
        else:
            self.batch_handler = None
    
    def cached_query(self, func):
        """
        Decorator for caching query results.
        
        Args:
            func: Function to decorate
        
        Returns:
            Decorated function with caching
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_caching or not self.cache:
                return func(*args, **kwargs)
            
            # Generate a cache key based on function name and arguments
            func_name = func.__name__
            args_str = json.dumps([str(arg) for arg in args[1:]], sort_keys=True)  # Skip self
            kwargs_str = json.dumps(kwargs, sort_keys=True)
            cache_key = f"{func_name}:{args_str}:{kwargs_str}"
            
            # Check if result is in cache
            cached_result = self.cache.get(cache_key, {})
            if cached_result:
                return cached_result
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Store result in cache
            self.cache.set(cache_key, {}, result)
            
            return result
        
        return wrapper
    
    def optimize_query(self, query: str, params: Dict[str, Any]) -> str:
        """
        Optimize a SQL query for better performance.
        
        Args:
            query: SQL query to optimize
            params: Query parameters
            
        Returns:
            Optimized SQL query
        """
        # Basic query optimizations
        optimized_query = query
        
        # Add PRAGMA for optimization if not already present
        if "PRAGMA" not in optimized_query.upper():
            optimized_query = "PRAGMA threads=4;\n" + optimized_query
        
        # Add index hint for improved performance if filtering on a specific column
        if "WHERE" in optimized_query.upper():
            # Check for common filtering columns and add index hints
            for col in ["model_id", "hardware_id", "hardware_type", "validation_timestamp"]:
                if col in optimized_query and f"/* {col}_idx */" not in optimized_query:
                    optimized_query = optimized_query.replace(
                        f"{col} =", f"{col} /* {col}_idx */ ="
                    )
        
        return optimized_query
    
    def create_indexes(self) -> None:
        """
        Create indexes on commonly queried columns to improve performance.
        """
        if not self.db_integration or not hasattr(self.db_integration, 'db_api'):
            logger.error("Cannot create indexes: No database connection")
            return
        
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Define indexes to create
            indexes = [
                # validation_results indexes
                "CREATE INDEX IF NOT EXISTS validation_timestamp_idx ON validation_results(validation_timestamp)",
                "CREATE INDEX IF NOT EXISTS simulation_result_id_idx ON validation_results(simulation_result_id)",
                "CREATE INDEX IF NOT EXISTS hardware_result_id_idx ON validation_results(hardware_result_id)",
                
                # simulation_results indexes
                "CREATE INDEX IF NOT EXISTS sim_model_id_idx ON simulation_results(model_id)",
                "CREATE INDEX IF NOT EXISTS sim_hardware_id_idx ON simulation_results(hardware_id)",
                "CREATE INDEX IF NOT EXISTS sim_timestamp_idx ON simulation_results(timestamp)",
                "CREATE INDEX IF NOT EXISTS sim_batch_size_idx ON simulation_results(batch_size)",
                
                # hardware_results indexes
                "CREATE INDEX IF NOT EXISTS hw_model_id_idx ON hardware_results(model_id)",
                "CREATE INDEX IF NOT EXISTS hw_hardware_id_idx ON hardware_results(hardware_id)",
                "CREATE INDEX IF NOT EXISTS hw_timestamp_idx ON hardware_results(timestamp)",
                "CREATE INDEX IF NOT EXISTS hw_batch_size_idx ON hardware_results(batch_size)",
                
                # calibration_history indexes
                "CREATE INDEX IF NOT EXISTS cal_hardware_type_idx ON calibration_history(hardware_type)",
                "CREATE INDEX IF NOT EXISTS cal_model_type_idx ON calibration_history(model_type)",
                "CREATE INDEX IF NOT EXISTS cal_timestamp_idx ON calibration_history(timestamp)",
                
                # drift_detection indexes
                "CREATE INDEX IF NOT EXISTS drift_hardware_type_idx ON drift_detection(hardware_type)",
                "CREATE INDEX IF NOT EXISTS drift_model_type_idx ON drift_detection(model_type)",
                "CREATE INDEX IF NOT EXISTS drift_timestamp_idx ON drift_detection(timestamp)",
                "CREATE INDEX IF NOT EXISTS drift_is_significant_idx ON drift_detection(is_significant)"
            ]
            
            # Create indexes
            for index_sql in indexes:
                conn.execute(index_sql)
            
            logger.info(f"Created {len(indexes)} indexes on database tables")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def analyze_tables(self) -> None:
        """
        Run ANALYZE on tables to update statistics for the query optimizer.
        """
        if not self.db_integration or not hasattr(self.db_integration, 'db_api'):
            logger.error("Cannot analyze tables: No database connection")
            return
        
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Get list of tables
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            # Run ANALYZE on each table
            for table in tables:
                table_name = table[0]
                logger.info(f"Running ANALYZE on table {table_name}")
                conn.execute(f"ANALYZE {table_name}")
            
            logger.info(f"Analyzed {len(tables)} tables to update statistics")
        except Exception as e:
            logger.error(f"Error analyzing tables: {e}")
    
    def batch_insert_validation_results(self, results: List[ValidationResult]) -> None:
        """
        Insert multiple validation results in a batch operation.
        
        Args:
            results: List of ValidationResult objects to insert
        """
        if not self.db_integration or not self.batch_handler:
            logger.error("Cannot batch insert: No database connection")
            return
        
        try:
            from data.duckdb.simulation_validation.core.schema import SimulationValidationSchema as schema
            
            for val_result in results:
                # Prepare simulation result record
                sim_record = schema.simulation_result_to_db_dict(val_result.simulation_result)
                sim_id = sim_record["id"]
                
                # Prepare hardware result record
                hw_record = schema.hardware_result_to_db_dict(val_result.hardware_result)
                hw_id = hw_record["id"]
                
                # Prepare validation result record
                val_record = schema.validation_result_to_db_dict(val_result, sim_id, hw_id)
                
                # Add operations to batch
                self.batch_handler.add_operation(
                    """
                    INSERT INTO simulation_results 
                    VALUES (:id, :model_id, :hardware_id, :batch_size, :precision, 
                            :timestamp, :simulation_version, :additional_metadata,
                            :throughput_items_per_second, :average_latency_ms, 
                            :memory_peak_mb, :power_consumption_w, 
                            :initialization_time_ms, :warmup_time_ms, 
                            CURRENT_TIMESTAMP)
                    """,
                    sim_record
                )
                
                self.batch_handler.add_operation(
                    """
                    INSERT INTO hardware_results 
                    VALUES (:id, :model_id, :hardware_id, :batch_size, :precision, 
                            :timestamp, :hardware_details, :test_environment, :additional_metadata,
                            :throughput_items_per_second, :average_latency_ms, 
                            :memory_peak_mb, :power_consumption_w, 
                            :initialization_time_ms, :warmup_time_ms, 
                            CURRENT_TIMESTAMP)
                    """,
                    hw_record
                )
                
                self.batch_handler.add_operation(
                    """
                    INSERT INTO validation_results 
                    VALUES (:id, :simulation_result_id, :hardware_result_id, 
                            :validation_timestamp, :validation_version, 
                            :metrics_comparison, :additional_metrics,
                            :overall_accuracy_score, :throughput_mape, 
                            :latency_mape, :memory_mape, :power_mape, 
                            CURRENT_TIMESTAMP)
                    """,
                    val_record
                )
            
            # Execute any remaining operations
            self.batch_handler.execute()
            
            logger.info(f"Batch inserted {len(results)} validation results")
        except Exception as e:
            logger.error(f"Error batch inserting validation results: {e}")
    
    def get_validation_results_optimized(
        self,
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Optimized version of get_validation_results with caching.
        
        Args:
            hardware_id: Filter by hardware ID
            model_id: Filter by model ID
            batch_size: Filter by batch size
            precision: Filter by precision
            start_date: Filter by validation date (start)
            end_date: Filter by validation date (end)
            limit: Maximum number of results to return
            use_cache: Whether to use the query cache
            
        Returns:
            List of validation result records
        """
        if not self.db_integration or not hasattr(self.db_integration, 'db_api'):
            logger.error("Cannot get validation results: No database connection")
            return []
        
        # Check cache if enabled
        if use_cache and self.enable_caching and self.cache:
            cache_key = f"get_validation_results:{hardware_id}:{model_id}:{batch_size}:{precision}:{start_date}:{end_date}:{limit}"
            cached_result = self.cache.get(cache_key, {})
            if cached_result:
                return cached_result
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_id:
                conditions.append("hr.hardware_id /* hw_hardware_id_idx */ = :hardware_id")
                params["hardware_id"] = hardware_id
            
            if model_id:
                conditions.append("sr.model_id /* sim_model_id_idx */ = :model_id")
                params["model_id"] = model_id
            
            if batch_size:
                conditions.append("hr.batch_size /* hw_batch_size_idx */ = :batch_size")
                params["batch_size"] = batch_size
            
            if precision:
                conditions.append("hr.precision = :precision")
                params["precision"] = precision
            
            if start_date:
                conditions.append("vr.validation_timestamp /* validation_timestamp_idx */ >= :start_date")
                params["start_date"] = start_date
            
            if end_date:
                conditions.append("vr.validation_timestamp /* validation_timestamp_idx */ <= :end_date")
                params["end_date"] = end_date
            
            # Build query with optimizations
            query = """
                PRAGMA threads=4; -- Use multiple threads for query execution
                SELECT 
                    vr.id as validation_id,
                    sr.id as simulation_id,
                    hr.id as hardware_id,
                    vr.validation_timestamp,
                    vr.validation_version,
                    vr.metrics_comparison,
                    vr.additional_metrics,
                    vr.overall_accuracy_score,
                    vr.throughput_mape,
                    vr.latency_mape,
                    vr.memory_mape,
                    vr.power_mape,
                    sr.model_id as model_id,
                    sr.hardware_id as hardware_type,
                    sr.batch_size,
                    sr.precision,
                    sr.simulation_version,
                    hr.hardware_details,
                    hr.test_environment
                FROM validation_results vr
                JOIN simulation_results sr ON vr.simulation_result_id /* simulation_result_id_idx */ = sr.id
                JOIN hardware_results hr ON vr.hardware_result_id /* hardware_result_id_idx */ = hr.id
            """
            
            if conditions:
                query += f" WHERE {' AND '.join(conditions)}"
            
            query += """
                ORDER BY vr.validation_timestamp DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute query
            conn = self.db_integration.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            validation_results = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                validation_results.append(record)
            
            # Store in cache if enabled
            if use_cache and self.enable_caching and self.cache:
                self.cache.set(cache_key, {}, validation_results)
            
            logger.info(f"Retrieved {len(validation_results)} validation results (optimized)")
            return validation_results
        except Exception as e:
            logger.error(f"Failed to get validation results: {e}")
            return []
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to store the backup (if None, auto-generate)
            
        Returns:
            Path to the backup file
        """
        if not os.path.exists(self.db_path):
            logger.error(f"Cannot backup database: File {self.db_path} does not exist")
            return None
        
        try:
            # Generate backup path if not provided
            if backup_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(os.path.dirname(self.db_path), "backups")
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f"{os.path.basename(self.db_path)}.{timestamp}.bak")
            
            # Create backup
            if hasattr(self.db_integration, 'db_api') and self.db_integration.db_api:
                # Close any open connections
                self.db_integration.db_api._get_connection().close()
            
            # Copy the database file
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Created database backup at {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return None
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore the database from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(backup_path):
            logger.error(f"Cannot restore database: Backup file {backup_path} does not exist")
            return False
        
        try:
            # Close any open connections
            if hasattr(self.db_integration, 'db_api') and self.db_integration.db_api:
                self.db_integration.db_api._get_connection().close()
            
            # Create a backup of the current database
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            current_backup = f"{self.db_path}.{timestamp}.prerestorebak"
            shutil.copy2(self.db_path, current_backup)
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Restored database from backup at {backup_path}")
            logger.info(f"Original database backed up at {current_backup}")
            
            # Reconnect to the database
            self.db_integration = SimulationValidationDBIntegration(db_path=self.db_path)
            
            return True
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            return False
    
    def optimize_database(self) -> bool:
        """
        Perform database optimization including vacuum and analyze.
        
        Returns:
            True if successful, False otherwise
        """
        if not hasattr(self.db_integration, 'db_api') or not self.db_integration.db_api:
            logger.error("Cannot optimize database: No database connection")
            return False
        
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Clear cache
            if self.enable_caching and self.cache:
                self.cache.invalidate()
            
            # Run optimization commands
            logger.info("Starting database optimization...")
            
            # Create indexes
            self.create_indexes()
            
            # Analyze tables
            self.analyze_tables()
            
            # Vacuum database to reclaim space
            logger.info("Vacuuming database...")
            conn.execute("VACUUM")
            
            # Run pragma optimizations
            logger.info("Applying database optimizations...")
            conn.execute("PRAGMA optimize")
            
            logger.info("Database optimization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        if not hasattr(self.db_integration, 'db_api') or not self.db_integration.db_api:
            logger.error("Cannot get database stats: No database connection")
            return {}
        
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Get table sizes
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_stats = {}
            
            for table in tables:
                table_name = table[0]
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                size_result = conn.execute(count_query).fetchone()
                size = size_result[0] if size_result else 0
                table_stats[table_name] = size
            
            # Get database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            # Get cache stats if enabled
            cache_stats = self.cache.get_stats() if self.enable_caching and self.cache else {}
            
            # Get index information
            indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            index_list = [idx[0] for idx in indexes]
            
            return {
                "database_path": self.db_path,
                "file_size_bytes": db_size,
                "file_size_mb": round(db_size / (1024 * 1024), 2),
                "tables": table_stats,
                "total_records": sum(table_stats.values()),
                "indexes": index_list,
                "index_count": len(index_list),
                "caching_enabled": self.enable_caching,
                "cache_stats": cache_stats
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_records(
        self,
        older_than_days: int,
        tables: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Clean up old records from the database.
        
        Args:
            older_than_days: Remove records older than this many days
            tables: List of tables to clean up (if None, use all timestamp-containing tables)
            dry_run: If True, only count records without deleting
            
        Returns:
            Dictionary with count of deleted records per table
        """
        if not hasattr(self.db_integration, 'db_api') or not self.db_integration.db_api:
            logger.error("Cannot clean up old records: No database connection")
            return {}
        
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Default tables if not specified
            if tables is None:
                tables = [
                    "validation_results",
                    "simulation_results",
                    "hardware_results",
                    "calibration_history",
                    "drift_detection"
                ]
            
            # Define timestamp columns for each table
            timestamp_columns = {
                "validation_results": "validation_timestamp",
                "simulation_results": "timestamp",
                "hardware_results": "timestamp",
                "calibration_history": "timestamp",
                "drift_detection": "timestamp"
            }
            
            # Calculate cutoff date
            cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=older_than_days)).isoformat()
            
            # Track record counts
            results = {}
            
            for table in tables:
                if table not in timestamp_columns:
                    logger.warning(f"Skipping table {table}: No timestamp column defined")
                    continue
                
                # Get timestamp column
                timestamp_col = timestamp_columns[table]
                
                # Count records to delete
                count_query = f"SELECT COUNT(*) FROM {table} WHERE {timestamp_col} < ?"
                count_result = conn.execute(count_query, [cutoff_date]).fetchone()
                count = count_result[0] if count_result else 0
                
                results[table] = {"count": count, "deleted": 0}
                
                # Skip if no records to delete
                if count == 0:
                    logger.info(f"No records to delete from {table}")
                    continue
                
                if not dry_run:
                    # Delete records
                    delete_query = f"DELETE FROM {table} WHERE {timestamp_col} < ?"
                    conn.execute(delete_query, [cutoff_date])
                    conn.commit()
                    
                    # Verify deletion
                    verify_count = conn.execute(count_query, [cutoff_date]).fetchone()
                    verify_count = verify_count[0] if verify_count else 0
                    
                    deleted_count = count - verify_count
                    results[table]["deleted"] = deleted_count
                    
                    logger.info(f"Deleted {deleted_count} records from {table}")
                else:
                    logger.info(f"Would delete {count} records from {table} (dry run)")
            
            if not dry_run:
                # Invalidate cache if enabled
                if self.enable_caching and self.cache:
                    self.cache.invalidate()
            
            return results
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """
        Clear the query cache.
        """
        if self.enable_caching and self.cache:
            self.cache.invalidate()
            logger.info("Query cache cleared")
        else:
            logger.warning("Query caching is not enabled")

    def get_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive database performance metrics.
        
        This method collects various performance metrics from the database including:
        - Query time
        - Storage size and growth
        - Index efficiency
        - Cache performance
        - Read/write efficiency
        - Memory usage
        - Compression ratio
        
        Returns:
            Dictionary mapping metric names to metric information including value,
            history, status, and other relevant information
        """
        metrics = {}
        
        if not hasattr(self.db_integration, 'db_api') or not self.db_integration.db_api:
            logger.error("Cannot get performance metrics: No database connection")
            return metrics
        
        try:
            # Get database stats as base information
            db_stats = self.get_database_stats()
            
            # Get connection for queries
            conn = self.db_integration.db_api._get_connection()
            
            # Query Time Metrics
            metrics["query_time"] = self._get_query_time_metrics(conn)
            
            # Storage Size Metrics
            metrics["storage_size"] = self._get_storage_metrics(db_stats)
            
            # Index Efficiency Metrics
            metrics["index_efficiency"] = self._get_index_efficiency_metrics(conn)
            
            # Read Efficiency Metrics
            metrics["read_efficiency"] = self._get_read_efficiency_metrics()
            
            # Write Efficiency Metrics
            metrics["write_efficiency"] = self._get_write_efficiency_metrics()
            
            # Vacuum Status Metrics
            metrics["vacuum_status"] = self._get_vacuum_status_metrics(conn)
            
            # Compression Ratio Metrics
            metrics["compression_ratio"] = self._get_compression_metrics(db_stats)
            
            # Cache Performance Metrics (if enabled)
            if self.enable_caching and self.cache:
                metrics["cache_performance"] = self._get_cache_performance_metrics(db_stats)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                "error": {
                    "value": str(e),
                    "status": "error",
                    "unit": "error",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }
    
    def _get_query_time_metrics(self, conn) -> Dict[str, Any]:
        """Get query time performance metrics."""
        try:
            # Measure query time for a standard benchmark query
            start_time = time.time()
            
            # Use a typical query for benchmarking
            conn.execute("""
                SELECT COUNT(*) FROM validation_results 
                JOIN simulation_results ON validation_results.simulation_result_id = simulation_results.id
                JOIN hardware_results ON validation_results.hardware_result_id = hardware_results.id
            """)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            # Determine status based on query time
            status = "good"
            if query_time_ms > 500:
                status = "warning"
            elif query_time_ms > 1000:
                status = "error"
            
            # Use a stored history or start one
            history_file = os.path.join(os.path.dirname(self.db_path), ".query_time_history.json")
            history = []
            
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                        history = history_data.get("history", [])
                        # Keep only the last 20 entries
                        history = history[-19:] if len(history) > 19 else history
                except Exception as e:
                    logger.warning(f"Could not read query time history: {e}")
            
            # Calculate previous value and change
            previous_value = history[-1] if history else None
            change_pct = ((query_time_ms - previous_value) / previous_value * 100) if previous_value else 0
            
            # Add current value to history
            history.append(query_time_ms)
            
            # Save updated history
            try:
                with open(history_file, 'w') as f:
                    json.dump({"history": history}, f)
            except Exception as e:
                logger.warning(f"Could not save query time history: {e}")
            
            return {
                "value": query_time_ms,
                "previous_value": previous_value,
                "change_pct": change_pct,
                "unit": "ms",
                "status": status,
                "history": history,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting query time metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "ms",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _get_storage_metrics(self, db_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get storage size and growth metrics."""
        try:
            # Get current size from stats
            current_size = db_stats.get("file_size_bytes", 0)
            
            # Use a stored history or start one
            history_file = os.path.join(os.path.dirname(self.db_path), ".storage_size_history.json")
            history = []
            
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                        history = history_data.get("history", [])
                        # Keep only the last 20 entries
                        history = history[-19:] if len(history) > 19 else history
                except Exception as e:
                    logger.warning(f"Could not read storage size history: {e}")
            
            # Calculate previous value and change
            previous_value = history[-1] if history else None
            change_pct = ((current_size - previous_value) / previous_value * 100) if previous_value else 0
            
            # Add current value to history
            history.append(current_size)
            
            # Save updated history
            try:
                with open(history_file, 'w') as f:
                    json.dump({"history": history}, f)
            except Exception as e:
                logger.warning(f"Could not save storage size history: {e}")
            
            # Determine status based on growth rate
            status = "good"
            if change_pct > 10:
                status = "warning"
            elif change_pct > 25:
                status = "error"
            
            return {
                "value": current_size,
                "previous_value": previous_value,
                "change_pct": change_pct,
                "unit": "bytes",
                "status": status,
                "history": history,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting storage metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "bytes",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _get_index_efficiency_metrics(self, conn) -> Dict[str, Any]:
        """Get index efficiency metrics."""
        try:
            # Check if EXPLAIN QUERY PLAN output mentions index usage
            query_plan = conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM validation_results 
                WHERE validation_timestamp > '2025-01-01'
                ORDER BY validation_timestamp DESC
                LIMIT 10
            """).fetchall()
            
            # Convert query plan to string for analysis
            plan_str = "\n".join([str(row) for row in query_plan])
            
            # Check if indexes are being used
            index_used = "INDEX" in plan_str.upper() or "USING INDEX" in plan_str.upper()
            
            # Get index count
            index_count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
            ).fetchone()[0]
            
            # Calculate an efficiency score (0-100)
            # This is a simple heuristic - real efficiency would require more complex analysis
            efficiency_score = 0
            if index_count > 0:
                if index_used:
                    efficiency_score = 100  # Good index usage
                else:
                    # If we have indexes but they're not being used, that's inefficient
                    efficiency_score = 50
            else:
                # No indexes exists
                efficiency_score = 25
            
            # Determine status based on efficiency score
            status = "good"
            if efficiency_score < 70:
                status = "warning"
            elif efficiency_score < 40:
                status = "error"
            
            return {
                "value": efficiency_score,
                "unit": "percent",
                "status": status,
                "index_count": index_count,
                "index_used": index_used,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting index efficiency metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "percent",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _get_read_efficiency_metrics(self) -> Dict[str, Any]:
        """Get read efficiency metrics."""
        try:
            # Benchmark read performance
            start_time = time.time()
            
            # Read a sample of validation results
            if hasattr(self.db_integration, 'get_validation_results'):
                self.db_integration.get_validation_results(limit=100)
            else:
                # Use our optimized method
                self.get_validation_results_optimized(limit=100)
            
            read_time_ms = (time.time() - start_time) * 1000
            
            # Calculate records per second
            records_per_second = 100 / (read_time_ms / 1000) if read_time_ms > 0 else 0
            
            # Determine status based on records per second
            status = "good"
            if records_per_second < 100:
                status = "warning"
            elif records_per_second < 50:
                status = "error"
            
            return {
                "value": records_per_second,
                "unit": "records/second",
                "status": status,
                "read_time_ms": read_time_ms,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting read efficiency metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "records/second",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _get_write_efficiency_metrics(self) -> Dict[str, Any]:
        """Get write efficiency metrics."""
        try:
            # Create a small temporary table for write testing
            conn = self.db_integration.db_api._get_connection()
            
            # Create test table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _perf_test_write (
                    id INTEGER PRIMARY KEY,
                    value TEXT,
                    timestamp TIMESTAMP
                )
            """)
            
            # Clear previous test data
            conn.execute("DELETE FROM _perf_test_write")
            
            # Generate test data
            test_data = [(i, f"test-{i}", datetime.datetime.now().isoformat()) for i in range(100)]
            
            # Measure write time
            start_time = time.time()
            
            # Use batch operation for the test
            batch_handler = BatchOperation(self.db_integration.db_api, batch_size=20)
            
            for i, value, timestamp in test_data:
                batch_handler.add_operation(
                    "INSERT INTO _perf_test_write VALUES (?, ?, ?)",
                    {"0": i, "1": value, "2": timestamp}
                )
            
            batch_handler.execute()
            
            write_time_ms = (time.time() - start_time) * 1000
            
            # Calculate records per second
            records_per_second = 100 / (write_time_ms / 1000) if write_time_ms > 0 else 0
            
            # Clean up test table
            conn.execute("DELETE FROM _perf_test_write")
            
            # Determine status based on records per second
            status = "good"
            if records_per_second < 100:
                status = "warning"
            elif records_per_second < 50:
                status = "error"
            
            return {
                "value": records_per_second,
                "unit": "records/second",
                "status": status,
                "write_time_ms": write_time_ms,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting write efficiency metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "records/second",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _get_vacuum_status_metrics(self, conn) -> Dict[str, Any]:
        """Get vacuum status metrics."""
        try:
            # Calculate approximate vacuum need based on free pages
            # This is a DuckDB-specific approach
            page_info = conn.execute("PRAGMA page_count, page_size, freelist_count").fetchone()
            
            if page_info and len(page_info) >= 3:
                page_count, page_size, freelist_count = page_info
                
                # Calculate free space percentage
                free_percent = (freelist_count / page_count * 100) if page_count > 0 else 0
                
                # Calculate vacuum status (100 is best - no vacuum needed)
                vacuum_status = 100 - free_percent
                
                # Determine status based on vacuum status
                status = "good"
                if vacuum_status < 80:
                    status = "warning"
                elif vacuum_status < 60:
                    status = "error"
                
                return {
                    "value": vacuum_status,
                    "unit": "percent",
                    "status": status,
                    "free_pages": freelist_count,
                    "total_pages": page_count,
                    "page_size": page_size,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                # Fallback if we can't get page info
                return {
                    "value": 50,  # Neutral value
                    "unit": "percent",
                    "status": "warning",
                    "timestamp": datetime.datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting vacuum status metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "percent",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _get_compression_metrics(self, db_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get compression ratio metrics."""
        try:
            # Estimate the theoretical uncompressed size based on record counts
            # This is a rough approximation
            tables = db_stats.get("tables", {})
            total_records = db_stats.get("total_records", 0)
            file_size = db_stats.get("file_size_bytes", 0)
            
            # Estimate average record size (higher values indicate more efficient storage)
            # Base compression ratio on assumption of ~500 bytes per record uncompressed
            estimated_uncompressed = total_records * 500
            compression_ratio = estimated_uncompressed / file_size if file_size > 0 else 0
            
            # Determine status based on compression ratio
            status = "good"
            if compression_ratio < 2:
                status = "warning"
            elif compression_ratio < 1:
                status = "error"
            
            return {
                "value": compression_ratio,
                "unit": "ratio",
                "status": status,
                "estimated_uncompressed_bytes": estimated_uncompressed,
                "actual_bytes": file_size,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting compression metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "ratio",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _get_cache_performance_metrics(self, db_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get cache performance metrics."""
        try:
            cache_stats = db_stats.get("cache_stats", {})
            
            hit_ratio = cache_stats.get("hit_ratio", 0)
            
            # Determine status based on hit ratio
            status = "good"
            if hit_ratio < 0.5:
                status = "warning"
            elif hit_ratio < 0.3:
                status = "error"
            
            return {
                "value": hit_ratio * 100,  # Convert to percentage
                "unit": "percent",
                "status": status,
                "hits": cache_stats.get("hits", 0),
                "misses": cache_stats.get("misses", 0),
                "cache_size": cache_stats.get("size", 0),
                "max_size": cache_stats.get("max_size", 0),
                "utilization": cache_stats.get("size", 0) / cache_stats.get("max_size", 1) * 100,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting cache performance metrics: {e}")
            return {
                "value": None,
                "status": "error",
                "unit": "percent",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def get_overall_status(self) -> str:
        """
        Get the overall status of the database performance.
        
        Returns:
            Status string: 'good', 'warning', or 'error'
        """
        try:
            # Get all metrics
            metrics = self.get_performance_metrics()
            
            # Count status types
            status_counts = {"good": 0, "warning": 0, "error": 0}
            
            for metric_name, metric_data in metrics.items():
                status = metric_data.get("status")
                if status in status_counts:
                    status_counts[status] += 1
            
            # Determine overall status
            if status_counts["error"] > 0:
                return "error"
            elif status_counts["warning"] > 0:
                return "warning"
            else:
                return "good"
        except Exception as e:
            logger.error(f"Error getting overall status: {e}")
            return "error"


def get_db_optimizer(
    db_path: str = "./benchmark_db.duckdb",
    enable_caching: bool = True,
    cache_size: int = 100,
    cache_ttl: int = 300,
    batch_size: int = 100
) -> DBPerformanceOptimizer:
    """
    Get an instance of the DBPerformanceOptimizer.
    
    Args:
        db_path: Path to the DuckDB database
        enable_caching: Whether to enable query caching
        cache_size: Maximum number of cached results to store
        cache_ttl: Time-to-live in seconds for cached results
        batch_size: Default batch size for batch operations
    
    Returns:
        DBPerformanceOptimizer instance
    """
    return DBPerformanceOptimizer(
        db_path=db_path,
        enable_caching=enable_caching,
        cache_size=cache_size,
        cache_ttl=cache_ttl,
        batch_size=batch_size
    )


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Database Performance Optimization Tool")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to the DuckDB database")
    parser.add_argument("--action", choices=["optimize", "backup", "restore", "stats", "cleanup", "create-indexes", "analyze"], 
                        default="optimize", help="Action to perform")
    parser.add_argument("--backup-path", help="Path for backup/restore operations")
    parser.add_argument("--days", type=int, default=90, help="Days threshold for cleanup operation")
    parser.add_argument("--tables", nargs="+", help="Tables for cleanup operation")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for cleanup operation")
    parser.add_argument("--disable-cache", action="store_true", help="Disable query caching")
    parser.add_argument("--cache-size", type=int, default=100, help="Query cache size")
    parser.add_argument("--cache-ttl", type=int, default=300, help="Query cache TTL in seconds")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch operation size")
    
    args = parser.parse_args()
    
    # Create optimizer instance
    optimizer = get_db_optimizer(
        db_path=args.db_path,
        enable_caching=not args.disable_cache,
        cache_size=args.cache_size,
        cache_ttl=args.cache_ttl,
        batch_size=args.batch_size
    )
    
    # Perform requested action
    if args.action == "optimize":
        logger.info("Optimizing database...")
        result = optimizer.optimize_database()
        logger.info(f"Optimization {'completed successfully' if result else 'failed'}")
    
    elif args.action == "backup":
        logger.info("Creating database backup...")
        backup_path = optimizer.backup_database(args.backup_path)
        if backup_path:
            logger.info(f"Backup created at: {backup_path}")
        else:
            logger.error("Backup failed")
    
    elif args.action == "restore":
        if not args.backup_path:
            logger.error("Backup path must be specified for restore operation")
            sys.exit(1)
        
        logger.info(f"Restoring database from backup: {args.backup_path}")
        result = optimizer.restore_database(args.backup_path)
        logger.info(f"Restore {'completed successfully' if result else 'failed'}")
    
    elif args.action == "stats":
        logger.info("Getting database statistics...")
        stats = optimizer.get_database_stats()
        
        # Print statistics in a readable format
        print("\nDatabase Statistics:")
        print(f"- Path: {stats.get('database_path', 'unknown')}")
        print(f"- Size: {stats.get('file_size_mb', 0)} MB")
        print(f"- Total Records: {stats.get('total_records', 0)}")
        print(f"- Number of Indexes: {stats.get('index_count', 0)}")
        
        if "tables" in stats:
            print("\nTable Record Counts:")
            for table, count in stats.get("tables", {}).items():
                print(f"- {table}: {count}")
        
        if "cache_stats" in stats:
            cache_stats = stats.get("cache_stats", {})
            print("\nCache Statistics:")
            print(f"- Enabled: {stats.get('caching_enabled', False)}")
            print(f"- Size: {cache_stats.get('size', 0)} / {cache_stats.get('max_size', 0)}")
            print(f"- Hits: {cache_stats.get('hits', 0)}")
            print(f"- Misses: {cache_stats.get('misses', 0)}")
            print(f"- Hit Ratio: {cache_stats.get('hit_ratio', 0):.2%}")
    
    elif args.action == "cleanup":
        logger.info(f"Cleaning up records older than {args.days} days...")
        results = optimizer.cleanup_old_records(args.days, args.tables, args.dry_run)
        
        # Print results
        print("\nCleanup Results:")
        for table, result in results.items():
            count = result.get("count", 0)
            deleted = result.get("deleted", 0)
            
            if args.dry_run:
                print(f"- {table}: Would delete {count} records")
            else:
                print(f"- {table}: Deleted {deleted}/{count} records")
    
    elif args.action == "create-indexes":
        logger.info("Creating database indexes...")
        optimizer.create_indexes()
        logger.info("Index creation completed")
    
    elif args.action == "analyze":
        logger.info("Analyzing database tables...")
        optimizer.analyze_tables()
        logger.info("Table analysis completed")