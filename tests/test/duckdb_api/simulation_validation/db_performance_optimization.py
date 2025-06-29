#!/usr/bin/env python3
"""
Database Performance Optimization Module for the Simulation Accuracy and Validation Framework.

This module provides performance enhancements for database operations, including:
- Query optimization for large datasets
- Batch operations for improved efficiency
- Query caching for frequently accessed data
- Database maintenance utilities
- Database backup and restore functionality
"""

import os
import sys
import json
import time
import datetime
import logging
import tempfile
import threading
import functools
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation_db_optimization")

# Import the database integration module
try:
    from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
except ImportError:
    logger.error("Failed to import SimulationValidationDBIntegration. Make sure duckdb_api is properly installed.")
    sys.exit(1)


class QueryCache:
    """
    Cache for database queries to improve performance for frequently accessed data.
    
    This cache implements a time-based invalidation strategy to ensure data freshness,
    while reducing database load for frequently accessed, infrequently changing data.
    """
    
    def __init__(self, ttl: int = 300, max_size: int = 100):
        """
        Initialize the query cache.
        
        Args:
            ttl: Time-to-live for cache entries in seconds (default: 5 minutes)
            max_size: Maximum number of entries in the cache (default: 100)
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}  # {query_hash: (result, timestamp)}
        self.lock = threading.RLock()
        logger.debug(f"Initialized QueryCache with TTL={ttl}s, max_size={max_size}")
    
    def _generate_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key for a query and parameters.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Cache key string
        """
        key_string = query
        if params:
            # Sort parameter keys to ensure consistent hash
            key_string += json.dumps(params, sort_keys=True)
        
        # Use SHA-256 hash for the key
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Get a result from the cache.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Cached result or None if not in cache or expired
        """
        key = self._generate_key(query, params)
        
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                # Check if the entry is still valid
                if time.time() - timestamp <= self.ttl:
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return result
                else:
                    # Remove expired entry
                    del self.cache[key]
                    logger.debug(f"Cache entry expired for query: {query[:50]}...")
            
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
    
    def set(self, query: str, params: Optional[Dict[str, Any]] = None, result: Any = None) -> None:
        """
        Store a result in the cache.
        
        Args:
            query: SQL query string
            params: Query parameters
            result: Query result to cache
        """
        key = self._generate_key(query, params)
        
        with self.lock:
            # If cache is full, remove oldest entry
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
                del self.cache[oldest_key]
                logger.debug("Cache full, removed oldest entry")
            
            # Store the result with current timestamp
            self.cache[key] = (result, time.time())
            logger.debug(f"Cached result for query: {query[:50]}...")
    
    def invalidate(self, table_name: Optional[str] = None) -> None:
        """
        Invalidate cache entries for a table or all entries.
        
        Args:
            table_name: Name of table to invalidate (or None for all)
        """
        with self.lock:
            if table_name:
                # Invalidate entries for specific table
                # This is a simple approach - we check if the table name appears in the query
                keys_to_remove = []
                for key, (result, timestamp) in self.cache.items():
                    # If the result contains metadata about the query, check for table name
                    if hasattr(result, 'query') and table_name.lower() in result.query.lower():
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.cache[key]
                
                logger.debug(f"Invalidated cache entries for table: {table_name}")
            else:
                # Invalidate all entries
                self.cache.clear()
                logger.debug("Invalidated all cache entries")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            logger.debug("Cleared all cache entries")


class BatchOperations:
    """
    Implements efficient batch operations for database interactions.
    
    This class optimizes database operations when dealing with large volumes of data
    by using batched inserts, updates, and deletes.
    """
    
    def __init__(self, db_integration: SimulationValidationDBIntegration, batch_size: int = 100):
        """
        Initialize the batch operations handler.
        
        Args:
            db_integration: Reference to the DB integration class
            batch_size: Number of records to process in each batch
        """
        self.db_integration = db_integration
        self.batch_size = batch_size
        logger.debug(f"Initialized BatchOperations with batch_size={batch_size}")
    
    def batch_insert(
        self, 
        table_name: str, 
        records: List[Dict[str, Any]], 
        columns: Optional[List[str]] = None
    ) -> int:
        """
        Insert multiple records in batches.
        
        Args:
            table_name: Name of the table
            records: List of record dictionaries
            columns: List of column names (if None, uses keys from first record)
            
        Returns:
            Number of records inserted
        """
        if not records:
            logger.warning("No records to insert")
            return 0
        
        # Get column names if not provided
        if columns is None:
            columns = list(records[0].keys())
        
        total_inserted = 0
        
        # Process in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i+self.batch_size]
            
            # Build query
            placeholders = ", ".join(["?"] * len(columns))
            columns_str = ", ".join(columns)
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES "
            
            # Add values for each record in the batch
            values = []
            for record in batch:
                record_values = [record.get(col) for col in columns]
                values.extend(record_values)
                query += f"({placeholders}), "
            
            # Remove trailing comma and space
            query = query[:-2]
            
            try:
                # Execute the batch insert
                conn = self.db_integration.db_api._get_connection()
                conn.execute(query, values)
                conn.commit()
                
                total_inserted += len(batch)
                logger.debug(f"Inserted batch of {len(batch)} records into {table_name}")
            except Exception as e:
                logger.error(f"Error inserting batch into {table_name}: {e}")
                if self.db_integration:
                    self.db_integration.db_api._get_connection().rollback()
                raise
        
        logger.info(f"Inserted total of {total_inserted} records into {table_name}")
        return total_inserted
    
    def batch_update(
        self, 
        table_name: str, 
        records: List[Dict[str, Any]], 
        id_column: str,
        update_columns: List[str]
    ) -> int:
        """
        Update multiple records in batches.
        
        Args:
            table_name: Name of the table
            records: List of record dictionaries
            id_column: Name of the ID column for identifying records
            update_columns: List of column names to update
            
        Returns:
            Number of records updated
        """
        if not records:
            logger.warning("No records to update")
            return 0
        
        total_updated = 0
        
        # Process in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i+self.batch_size]
            
            try:
                # For DuckDB, we can use the more efficient CASE-based approach for batch updates
                case_statements = []
                id_values = []
                
                for column in update_columns:
                    # Build CASE statement for this column
                    case_stmt = f"{column} = CASE {id_column} "
                    
                    # Add each record's value for this column
                    for record in batch:
                        case_stmt += f"WHEN ? THEN ? "
                        id_values.append(record[id_column])
                        id_values.append(record[column])
                    
                    case_stmt += "ELSE {column} END"
                    case_statements.append(case_stmt)
                
                # Build the full query
                set_clause = ", ".join(case_statements)
                id_list = ", ".join(["?"] * len(batch))
                
                # Extract ID values for the WHERE clause
                where_values = [record[id_column] for record in batch]
                
                query = f"UPDATE {table_name} SET {set_clause} WHERE {id_column} IN ({id_list})"
                
                # Execute the batch update
                conn = self.db_integration.db_api._get_connection()
                conn.execute(query, id_values + where_values)
                conn.commit()
                
                total_updated += len(batch)
                logger.debug(f"Updated batch of {len(batch)} records in {table_name}")
            except Exception as e:
                logger.error(f"Error updating batch in {table_name}: {e}")
                if self.db_integration:
                    self.db_integration.db_api._get_connection().rollback()
                raise
        
        logger.info(f"Updated total of {total_updated} records in {table_name}")
        return total_updated
    
    def batch_delete(
        self, 
        table_name: str, 
        id_column: str,
        id_values: List[Any]
    ) -> int:
        """
        Delete multiple records in batches.
        
        Args:
            table_name: Name of the table
            id_column: Name of the ID column for identifying records
            id_values: List of ID values to delete
            
        Returns:
            Number of records deleted
        """
        if not id_values:
            logger.warning("No records to delete")
            return 0
        
        total_deleted = 0
        
        # Process in batches
        for i in range(0, len(id_values), self.batch_size):
            batch = id_values[i:i+self.batch_size]
            
            try:
                # Build placeholders for the IN clause
                placeholders = ", ".join(["?"] * len(batch))
                
                query = f"DELETE FROM {table_name} WHERE {id_column} IN ({placeholders})"
                
                # Execute the batch delete
                conn = self.db_integration.db_api._get_connection()
                conn.execute(query, batch)
                conn.commit()
                
                total_deleted += len(batch)
                logger.debug(f"Deleted batch of {len(batch)} records from {table_name}")
            except Exception as e:
                logger.error(f"Error deleting batch from {table_name}: {e}")
                if self.db_integration:
                    self.db_integration.db_api._get_connection().rollback()
                raise
        
        logger.info(f"Deleted total of {total_deleted} records from {table_name}")
        return total_deleted


class QueryOptimizer:
    """
    Provides query optimization for large datasets.
    
    This class implements various optimizations for database queries when
    dealing with large volumes of data, including indexing, query rewriting,
    and execution plan analysis.
    """
    
    def __init__(self, db_integration: SimulationValidationDBIntegration):
        """
        Initialize the query optimizer.
        
        Args:
            db_integration: Reference to the DB integration class
        """
        self.db_integration = db_integration
        self._indexes_created = False
        logger.debug("Initialized QueryOptimizer")
    
    def create_indexes(self) -> None:
        """Create indexes for commonly queried columns."""
        if self._indexes_created:
            logger.debug("Indexes already created")
            return
        
        try:
            # Define indexes to create
            indexes = [
                # simulation_results indexes
                "CREATE INDEX IF NOT EXISTS idx_sim_model_hw ON simulation_results(model_id, hardware_id)",
                "CREATE INDEX IF NOT EXISTS idx_sim_timestamp ON simulation_results(timestamp)",
                
                # hardware_results indexes
                "CREATE INDEX IF NOT EXISTS idx_hw_model_hw ON hardware_results(model_id, hardware_id)",
                "CREATE INDEX IF NOT EXISTS idx_hw_timestamp ON hardware_results(timestamp)",
                
                # validation_results indexes
                "CREATE INDEX IF NOT EXISTS idx_val_sim_hw ON validation_results(simulation_result_id, hardware_result_id)",
                "CREATE INDEX IF NOT EXISTS idx_val_timestamp ON validation_results(validation_timestamp)",
                
                # calibration_history indexes
                "CREATE INDEX IF NOT EXISTS idx_cal_hw_model ON calibration_history(hardware_type, model_type)",
                "CREATE INDEX IF NOT EXISTS idx_cal_timestamp ON calibration_history(timestamp)",
                
                # drift_detection indexes
                "CREATE INDEX IF NOT EXISTS idx_drift_hw_model ON drift_detection(hardware_type, model_type)",
                "CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_detection(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_drift_significant ON drift_detection(is_significant)"
            ]
            
            # Create each index
            conn = self.db_integration.db_api._get_connection()
            for idx_sql in indexes:
                conn.execute(idx_sql)
            
            # Create covering indexes for common queries
            covering_indexes = [
                """CREATE INDEX IF NOT EXISTS idx_val_overall_metrics ON validation_results(
                    simulation_result_id, hardware_result_id, validation_timestamp,
                    overall_accuracy_score, throughput_mape, latency_mape,
                    memory_mape, power_mape
                )"""
            ]
            
            for idx_sql in covering_indexes:
                conn.execute(idx_sql)
            
            conn.commit()
            self._indexes_created = True
            logger.info("Created database indexes for performance optimization")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            if self.db_integration:
                self.db_integration.db_api._get_connection().rollback()
    
    def analyze_tables(self) -> None:
        """Analyze tables to update statistics for the query optimizer."""
        try:
            tables = [
                "simulation_results",
                "hardware_results",
                "validation_results",
                "calibration_history",
                "drift_detection"
            ]
            
            conn = self.db_integration.db_api._get_connection()
            for table in tables:
                # DuckDB doesn't have a direct ANALYZE command, but we can force statistics
                # collection by running a query that touches all rows
                conn.execute(f"SELECT COUNT(*) FROM {table}")
            
            logger.info("Analyzed database tables for query optimization")
        except Exception as e:
            logger.error(f"Error analyzing tables: {e}")
    
    def optimize_query(self, query: str) -> str:
        """
        Optimize a SQL query for better performance.
        
        Args:
            query: Original SQL query
            
        Returns:
            Optimized SQL query
        """
        # Ensure indexes are created
        if not self._indexes_created:
            self.create_indexes()
        
        # Apply query optimizations
        optimized_query = query
        
        # Optimization 1: Add LIMIT to queries without one
        if "LIMIT" not in optimized_query.upper() and "SELECT" in optimized_query.upper():
            optimized_query += " LIMIT 10000"
        
        # Optimization 2: Add query hints for index usage where applicable
        # Note: DuckDB automatically uses indexes, so we don't need explicit hints
        
        # Optimization 3: Rewrite queries with count(*) to count(1) for performance
        optimized_query = optimized_query.replace("COUNT(*)", "COUNT(1)")
        
        # Optimization 4: Use EXPLAIN to check query plan if debug is enabled
        if logger.level <= logging.DEBUG:
            explain_query = f"EXPLAIN {optimized_query}"
            try:
                conn = self.db_integration.db_api._get_connection()
                result = conn.execute(explain_query).fetchall()
                explain_text = "\n".join([str(row[0]) for row in result])
                logger.debug(f"Query execution plan:\n{explain_text}")
            except Exception as e:
                logger.debug(f"Error getting query plan: {e}")
        
        return optimized_query
    
    def execute_optimized_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        cache: Optional[QueryCache] = None
    ) -> List[Tuple]:
        """
        Execute an optimized query with optional caching.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            use_cache: Whether to use query cache
            cache: Query cache instance (if None, creates temporary cache)
            
        Returns:
            Query results
        """
        # Check cache if enabled
        if use_cache:
            if cache is None:
                # Create temporary cache for this query
                cache = QueryCache()
            
            cached_result = cache.get(query, params)
            if cached_result is not None:
                return cached_result
        
        # Optimize the query
        optimized_query = self.optimize_query(query)
        
        # Execute the query
        try:
            conn = self.db_integration.db_api._get_connection()
            start_time = time.time()
            
            if params:
                result = conn.execute(optimized_query, params).fetchall()
            else:
                result = conn.execute(optimized_query).fetchall()
            
            execution_time = time.time() - start_time
            
            # Log query performance
            if execution_time > 1.0:  # Log slow queries
                logger.warning(f"Slow query ({execution_time:.2f}s): {optimized_query[:100]}...")
            else:
                logger.debug(f"Query executed in {execution_time:.4f}s: {optimized_query[:100]}...")
            
            # Cache the result if enabled
            if use_cache and cache is not None:
                cache.set(query, params, result)
            
            return result
        except Exception as e:
            logger.error(f"Error executing optimized query: {e}")
            raise
    
    def get_table_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for database tables.
        
        Returns:
            Dictionary of table statistics
        """
        try:
            tables = [
                "simulation_results",
                "hardware_results",
                "validation_results",
                "calibration_history",
                "drift_detection"
            ]
            
            stats = {}
            conn = self.db_integration.db_api._get_connection()
            
            for table in tables:
                table_stats = {}
                
                # Get row count
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                table_stats["row_count"] = result[0]
                
                # Get column names
                result = conn.execute(f"PRAGMA table_info({table})").fetchall()
                table_stats["columns"] = [row[1] for row in result]
                
                # Get index information
                result = conn.execute(f"PRAGMA index_list({table})").fetchall()
                table_stats["indexes"] = [row[1] for row in result]
                
                # Get approximate storage size (estimate based on row count and column types)
                # DuckDB doesn't have a direct way to get table size, so we're estimating
                size_estimate = table_stats["row_count"] * len(table_stats["columns"]) * 20  # rough estimate
                table_stats["estimated_size_bytes"] = size_estimate
                
                stats[table] = table_stats
            
            return stats
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            return {}


class DatabaseMaintenance:
    """
    Provides database maintenance utilities.
    
    This class implements tools for database maintenance, including:
    - Vacuum operations to reclaim space
    - Integrity checks
    - Cleanup of old data
    - Performance monitoring
    """
    
    def __init__(self, db_integration: SimulationValidationDBIntegration):
        """
        Initialize the database maintenance utilities.
        
        Args:
            db_integration: Reference to the DB integration class
        """
        self.db_integration = db_integration
        logger.debug("Initialized DatabaseMaintenance")
    
    def vacuum_database(self) -> None:
        """Vacuum the database to reclaim space and optimize storage."""
        try:
            start_time = time.time()
            conn = self.db_integration.db_api._get_connection()
            
            # DuckDB uses a simplified VACUUM syntax
            conn.execute("VACUUM")
            
            duration = time.time() - start_time
            logger.info(f"Vacuumed database in {duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
    
    def check_integrity(self) -> Dict[str, bool]:
        """
        Check database integrity.
        
        Returns:
            Dictionary with integrity check results
        """
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Run pragma integrity_check
            result = conn.execute("PRAGMA integrity_check").fetchall()
            
            integrity_ok = len(result) == 1 and result[0][0] == "ok"
            
            # Run foreign key check
            conn.execute("PRAGMA foreign_keys = ON")
            fk_check = conn.execute("PRAGMA foreign_key_check").fetchall()
            
            foreign_keys_ok = len(fk_check) == 0
            
            return {
                "integrity_check": integrity_ok,
                "foreign_key_check": foreign_keys_ok,
                "overall": integrity_ok and foreign_keys_ok
            }
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return {
                "integrity_check": False,
                "foreign_key_check": False,
                "overall": False,
                "error": str(e)
            }
    
    def cleanup_old_data(
        self, 
        retention_days: int = 365, 
        tables: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Remove old data from the database.
        
        Args:
            retention_days: Number of days to retain data
            tables: List of tables to clean up (default: all)
            
        Returns:
            Dictionary with number of rows deleted per table
        """
        if tables is None:
            tables = [
                "simulation_results",
                "hardware_results",
                "validation_results",
                "calibration_history",
                "drift_detection"
            ]
        
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=retention_days)).strftime('%Y-%m-%d')
        deleted_counts = {}
        
        try:
            conn = self.db_integration.db_api._get_connection()
            
            for table in tables:
                # Determine timestamp column name based on table
                if table == "validation_results":
                    timestamp_col = "validation_timestamp"
                else:
                    timestamp_col = "timestamp"
                
                # Check if the table has the expected timestamp column
                try:
                    # Delete old records
                    result = conn.execute(
                        f"DELETE FROM {table} WHERE {timestamp_col} < ?",
                        [cutoff_date]
                    )
                    conn.commit()
                    
                    # Get number of rows deleted (DuckDB doesn't support rowcount directly)
                    deleted_counts[table] = result.rows_changed if hasattr(result, 'rows_changed') else 0
                    
                    logger.info(f"Deleted {deleted_counts[table]} records from {table} older than {cutoff_date}")
                except Exception as e:
                    logger.error(f"Error cleaning up {table}: {e}")
                    conn.rollback()
                    deleted_counts[table] = -1
            
            return deleted_counts
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
            if self.db_integration and hasattr(self.db_integration.db_api, '_get_connection'):
                self.db_integration.db_api._get_connection().rollback()
            return {"error": str(e)}
    
    def rebuild_indexes(self) -> None:
        """Rebuild database indexes for improved performance."""
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Get all indexes
            indexes_query = """
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            """
            indexes = conn.execute(indexes_query).fetchall()
            
            for idx in indexes:
                index_name = idx[0]
                # Drop and recreate index
                try:
                    # Get the index creation SQL
                    create_sql = conn.execute(
                        f"SELECT sql FROM sqlite_master WHERE name = ?",
                        [index_name]
                    ).fetchone()[0]
                    
                    # Drop the index
                    conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                    
                    # Recreate the index
                    conn.execute(create_sql)
                    
                    logger.debug(f"Rebuilt index: {index_name}")
                except Exception as e:
                    logger.error(f"Error rebuilding index {index_name}: {e}")
            
            conn.commit()
            logger.info(f"Rebuilt {len(indexes)} database indexes")
        except Exception as e:
            logger.error(f"Error rebuilding indexes: {e}")
            if self.db_integration:
                self.db_integration.db_api._get_connection().rollback()
    
    def analyze_query_performance(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze the performance of a query.
        
        Args:
            query: SQL query to analyze
            params: Query parameters
            iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            conn = self.db_integration.db_api._get_connection()
            
            # Run EXPLAIN to get query plan
            explain_query = f"EXPLAIN {query}"
            explain_result = None
            
            try:
                if params:
                    explain_result = conn.execute(explain_query, params).fetchall()
                else:
                    explain_result = conn.execute(explain_query).fetchall()
                
                explain_text = "\n".join([str(row[0]) for row in explain_result])
            except Exception as e:
                logger.warning(f"Error getting query plan: {e}")
                explain_text = f"Error: {e}"
            
            # Measure execution time over multiple iterations
            execution_times = []
            for i in range(iterations):
                start_time = time.time()
                
                if params:
                    conn.execute(query, params).fetchall()
                else:
                    conn.execute(query).fetchall()
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            # Calculate statistics
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            return {
                "query": query,
                "explain_plan": explain_text,
                "iterations": iterations,
                "avg_execution_time": avg_time,
                "min_execution_time": min_time,
                "max_execution_time": max_time,
                "execution_times": execution_times
            }
        except Exception as e:
            logger.error(f"Error analyzing query performance: {e}")
            return {
                "query": query,
                "error": str(e)
            }
    
    def get_database_size(self) -> int:
        """
        Get the current size of the database file in bytes.
        
        Returns:
            Size of the database file in bytes
        """
        try:
            db_path = self.db_integration.db_path
            if os.path.exists(db_path):
                return os.path.getsize(db_path)
            else:
                logger.warning(f"Database file not found: {db_path}")
                return 0
        except Exception as e:
            logger.error(f"Error getting database size: {e}")
            return 0
    
    def monitor_database_performance(
        self, 
        report_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Monitor database performance and generate a report.
        
        Args:
            report_file: Path to save the report (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            conn = self.db_integration.db_api._get_connection()
            report = {}
            
            # Get database file size
            report["database_size_bytes"] = self.get_database_size()
            report["database_size_mb"] = report["database_size_bytes"] / (1024 * 1024)
            
            # Get table counts
            tables = [
                "simulation_results",
                "hardware_results",
                "validation_results",
                "calibration_history",
                "drift_detection"
            ]
            
            table_counts = {}
            for table in tables:
                try:
                    result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    table_counts[table] = result[0]
                except Exception as e:
                    table_counts[table] = f"Error: {e}"
            
            report["table_counts"] = table_counts
            
            # Test some common queries
            test_queries = [
                "SELECT COUNT(*) FROM validation_results",
                "SELECT AVG(overall_accuracy_score) FROM validation_results",
                "SELECT model_id, hardware_id, COUNT(*) FROM validation_results GROUP BY model_id, hardware_id LIMIT 10"
            ]
            
            query_performance = {}
            for query in test_queries:
                perf = self.analyze_query_performance(query, iterations=2)
                query_performance[query] = {
                    "avg_execution_time": perf["avg_execution_time"],
                    "min_execution_time": perf["min_execution_time"],
                    "max_execution_time": perf["max_execution_time"]
                }
            
            report["query_performance"] = query_performance
            
            # Check integrity
            report["integrity_check"] = self.check_integrity()
            
            # Include timestamp
            report["timestamp"] = datetime.datetime.now().isoformat()
            
            # Save report if requested
            if report_file:
                os.makedirs(os.path.dirname(os.path.abspath(report_file)), exist_ok=True)
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Saved performance report to {report_file}")
            
            return report
        except Exception as e:
            logger.error(f"Error monitoring database performance: {e}")
            return {"error": str(e)}


class DatabaseBackupManager:
    """
    Provides database backup and restore functionality.
    
    This class implements tools for backing up and restoring databases,
    including scheduled backups, compression, and verification.
    """
    
    def __init__(self, db_integration: SimulationValidationDBIntegration, backup_dir: str = "./backups"):
        """
        Initialize the database backup manager.
        
        Args:
            db_integration: Reference to the DB integration class
            backup_dir: Directory to store backups
        """
        self.db_integration = db_integration
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Initialized DatabaseBackupManager with backup_dir={backup_dir}")
    
    def create_backup(
        self, 
        backup_name: Optional[str] = None,
        compress: bool = True,
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Create a database backup.
        
        Args:
            backup_name: Custom name for the backup (default: timestamp)
            compress: Whether to compress the backup
            verify: Whether to verify the backup after creation
            
        Returns:
            Dictionary with backup information
        """
        try:
            # Ensure database is closed before backup
            self.db_integration.close()
            
            # Generate backup name if not provided
            if backup_name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                db_filename = os.path.basename(self.db_integration.db_path)
                backup_name = f"{db_filename}_backup_{timestamp}"
            
            # Create backup path
            backup_path = self.backup_dir / backup_name
            
            # Copy the database file
            try:
                shutil.copy2(self.db_integration.db_path, backup_path)
                logger.info(f"Created database backup at {backup_path}")
            except Exception as e:
                logger.error(f"Error copying database for backup: {e}")
                raise
            
            backup_info = {
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "original_db": self.db_integration.db_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "size_bytes": os.path.getsize(backup_path),
                "compressed": False
            }
            
            # Compress the backup if requested
            if compress:
                try:
                    import gzip
                    compressed_path = str(backup_path) + ".gz"
                    
                    with open(backup_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove the uncompressed backup
                    os.unlink(backup_path)
                    
                    backup_info["backup_path"] = compressed_path
                    backup_info["compressed"] = True
                    backup_info["compressed_size_bytes"] = os.path.getsize(compressed_path)
                    backup_info["compression_ratio"] = backup_info["size_bytes"] / backup_info["compressed_size_bytes"]
                    
                    logger.info(f"Compressed backup to {compressed_path}")
                except Exception as e:
                    logger.error(f"Error compressing backup: {e}")
                    # Continue with uncompressed backup
            
            # Verify the backup if requested
            if verify:
                verification_result = self.verify_backup(backup_info["backup_path"], compressed=backup_info["compressed"])
                backup_info["verified"] = verification_result["success"]
                backup_info["verification_details"] = verification_result
                
                if not verification_result["success"]:
                    logger.error(f"Backup verification failed: {verification_result['error']}")
            
            # Save backup metadata
            metadata_path = self.backup_dir / f"{backup_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info(f"Backup completed: {backup_info['backup_path']}")
            return backup_info
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Reopen the database connection
            try:
                self.db_integration.db_api = self.db_integration.db_api.__class__(
                    db_path=self.db_integration.db_path, 
                    debug=(logger.level <= logging.DEBUG)
                )
            except Exception as e:
                logger.error(f"Error reopening database after backup: {e}")
    
    def verify_backup(
        self, 
        backup_path: str,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Verify a database backup.
        
        Args:
            backup_path: Path to the backup file
            compressed: Whether the backup is compressed
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Create a temporary file for verification
            temp_db = tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False)
            temp_db_path = temp_db.name
            temp_db.close()
            
            try:
                # Decompress if needed
                if compressed:
                    try:
                        import gzip
                        with gzip.open(backup_path, 'rb') as f_in:
                            with open(temp_db_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Error decompressing backup: {e}"
                        }
                else:
                    shutil.copy(backup_path, temp_db_path)
                
                # Connect to the temporary database
                from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
                temp_db_api = BenchmarkDBAPI(db_path=temp_db_path)
                
                # Run some basic checks
                checks = []
                
                # Check if tables exist
                tables = ["simulation_results", "hardware_results", "validation_results"]
                for table in tables:
                    try:
                        result = temp_db_api._get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                        checks.append({
                            "check": f"Table {table} exists",
                            "result": "Pass",
                            "details": f"Table has {result[0]} rows"
                        })
                    except Exception as e:
                        checks.append({
                            "check": f"Table {table} exists",
                            "result": "Fail",
                            "details": str(e)
                        })
                
                # Check database integrity
                try:
                    integrity = temp_db_api._get_connection().execute("PRAGMA integrity_check").fetchall()
                    integrity_ok = len(integrity) == 1 and integrity[0][0] == "ok"
                    
                    checks.append({
                        "check": "Database integrity",
                        "result": "Pass" if integrity_ok else "Fail",
                        "details": str(integrity)
                    })
                except Exception as e:
                    checks.append({
                        "check": "Database integrity",
                        "result": "Fail",
                        "details": str(e)
                    })
                
                # Overall success is true if all checks passed
                success = all(check["result"] == "Pass" for check in checks)
                
                return {
                    "success": success,
                    "checks": checks,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            finally:
                # Clean up the temporary database
                if os.path.exists(temp_db_path):
                    os.unlink(temp_db_path)
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def restore_backup(
        self, 
        backup_path: str,
        target_path: Optional[str] = None,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Restore a database from backup.
        
        Args:
            backup_path: Path to the backup file
            target_path: Path to restore to (default: original path)
            compressed: Whether the backup is compressed
            
        Returns:
            Dictionary with restoration results
        """
        try:
            # Close the database connection
            self.db_integration.close()
            
            # If target_path is not specified, use the original database path
            if target_path is None:
                target_path = self.db_integration.db_path
            
            # Create a backup of the current database if it exists
            if os.path.exists(target_path):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                pre_restore_backup = f"{target_path}_pre_restore_{timestamp}"
                shutil.copy2(target_path, pre_restore_backup)
                logger.info(f"Created pre-restore backup at {pre_restore_backup}")
            
            # Restore the database
            if compressed:
                try:
                    import gzip
                    with gzip.open(backup_path, 'rb') as f_in:
                        with open(target_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    logger.error(f"Error decompressing backup for restore: {e}")
                    raise
            else:
                shutil.copy2(backup_path, target_path)
            
            logger.info(f"Restored database from {backup_path} to {target_path}")
            
            # Reopen the database connection
            try:
                self.db_integration.db_api = self.db_integration.db_api.__class__(
                    db_path=target_path, 
                    debug=(logger.level <= logging.DEBUG)
                )
                
                # Verify the restored database
                conn = self.db_integration.db_api._get_connection()
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                
                return {
                    "success": True,
                    "backup_path": backup_path,
                    "target_path": target_path,
                    "compressed": compressed,
                    "tables_restored": len(tables),
                    "tables": [t[0] for t in tables],
                    "timestamp": datetime.datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error reopening database after restore: {e}")
                return {
                    "success": False,
                    "error": f"Restore succeeded but failed to reopen database: {e}",
                    "backup_path": backup_path,
                    "target_path": target_path,
                    "timestamp": datetime.datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            # Try to reopen the database connection
            try:
                self.db_integration.db_api = self.db_integration.db_api.__class__(
                    db_path=self.db_integration.db_path, 
                    debug=(logger.level <= logging.DEBUG)
                )
            except:
                pass
                
            return {
                "success": False,
                "error": str(e),
                "backup_path": backup_path,
                "target_path": target_path,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        try:
            # Look for backup metadata files
            metadata_files = list(self.backup_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if the backup file exists
                    backup_exists = os.path.exists(metadata["backup_path"])
                    metadata["file_exists"] = backup_exists
                    
                    if backup_exists:
                        # Update the file size (it might have changed)
                        metadata["current_size_bytes"] = os.path.getsize(metadata["backup_path"])
                    
                    backups.append(metadata)
                except Exception as e:
                    logger.warning(f"Error reading backup metadata {metadata_file}: {e}")
                    # Add basic info for the backup
                    backup_name = metadata_file.stem.replace("_metadata", "")
                    backups.append({
                        "backup_name": backup_name,
                        "metadata_path": str(metadata_file),
                        "error": str(e),
                        "file_exists": False  # We don't know the actual backup path
                    })
            
            # Sort by timestamp (most recent first)
            backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return backups
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def delete_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Delete a backup.
        
        Args:
            backup_name: Name of the backup to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            # Find the backup metadata
            metadata_path = self.backup_dir / f"{backup_name}_metadata.json"
            
            if not metadata_path.exists():
                return {
                    "success": False,
                    "error": f"Backup metadata not found: {metadata_path}"
                }
            
            # Load the metadata to get the backup file path
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            backup_path = metadata["backup_path"]
            
            # Delete the backup file
            files_deleted = []
            
            if os.path.exists(backup_path):
                os.unlink(backup_path)
                files_deleted.append(backup_path)
            
            # Delete the metadata file
            os.unlink(metadata_path)
            files_deleted.append(str(metadata_path))
            
            logger.info(f"Deleted backup: {backup_name}")
            
            return {
                "success": True,
                "backup_name": backup_name,
                "files_deleted": files_deleted,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error deleting backup {backup_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "backup_name": backup_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def schedule_backup(
        self, 
        interval_hours: int = 24,
        keep_backups: int = 7,
        compress: bool = True
    ) -> Dict[str, Any]:
        """
        Schedule regular database backups.
        
        Args:
            interval_hours: Interval between backups in hours
            keep_backups: Number of backups to keep
            compress: Whether to compress backups
            
        Returns:
            Dictionary with scheduling information
        """
        # This method doesn't actually implement scheduling directly
        # It creates a script that can be run by a scheduler like cron
        
        try:
            # Create a backup script
            script_path = self.backup_dir / "schedule_backup.py"
            
            script_content = f"""#!/usr/bin/env python3
\"\"\"
Scheduled backup script for the Simulation Validation database.
Generated by DatabaseBackupManager on {datetime.datetime.now().isoformat()}

Configure this script to run using cron or another scheduler.
Example cron entry to run every {interval_hours} hours:

0 */{interval_hours} * * * /usr/bin/python3 {script_path}
\"\"\"

import os
import sys
import json
import datetime
import shutil
import glob
from pathlib import Path

# Configuration
DB_PATH = "{self.db_integration.db_path}"
BACKUP_DIR = "{self.backup_dir}"
COMPRESS = {compress}
KEEP_BACKUPS = {keep_backups}

def create_backup():
    \"\"\"Create a database backup.\"\"\"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    db_filename = os.path.basename(DB_PATH)
    backup_name = f"{{db_filename}}_backup_{{timestamp}}"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    
    # Make sure the backup directory exists
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Copy the database file
    shutil.copy2(DB_PATH, backup_path)
    print(f"Created database backup at {{backup_path}}")
    
    backup_info = {{
        "backup_name": backup_name,
        "backup_path": backup_path,
        "original_db": DB_PATH,
        "timestamp": datetime.datetime.now().isoformat(),
        "size_bytes": os.path.getsize(backup_path),
        "compressed": False
    }}
    
    # Compress the backup if requested
    if COMPRESS:
        try:
            import gzip
            compressed_path = backup_path + ".gz"
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove the uncompressed backup
            os.unlink(backup_path)
            
            backup_info["backup_path"] = compressed_path
            backup_info["compressed"] = True
            backup_info["compressed_size_bytes"] = os.path.getsize(compressed_path)
            
            print(f"Compressed backup to {{compressed_path}}")
        except Exception as e:
            print(f"Error compressing backup: {{e}}")
    
    # Save backup metadata
    metadata_path = os.path.join(BACKUP_DIR, f"{{backup_name}}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(backup_info, f, indent=2)
    
    return backup_info

def cleanup_old_backups():
    \"\"\"Delete old backups, keeping only the most recent ones.\"\"\"
    # Get all backup metadata files
    metadata_files = sorted(
        glob.glob(os.path.join(BACKUP_DIR, "*_metadata.json")),
        key=os.path.getmtime,
        reverse=True  # Most recent first
    )
    
    # Keep only the specified number of backups
    if len(metadata_files) > KEEP_BACKUPS:
        for metadata_file in metadata_files[KEEP_BACKUPS:]:
            try:
                # Read the metadata to get the backup file path
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                backup_path = metadata["backup_path"]
                
                # Delete the backup file if it exists
                if os.path.exists(backup_path):
                    os.unlink(backup_path)
                    print(f"Deleted old backup: {{backup_path}}")
                
                # Delete the metadata file
                os.unlink(metadata_file)
                print(f"Deleted metadata: {{metadata_file}}")
            except Exception as e:
                print(f"Error deleting old backup {{metadata_file}}: {{e}}")

def main():
    \"\"\"Main function to create backup and clean up old ones.\"\"\"
    print(f"Running scheduled backup at {{datetime.datetime.now().isoformat()}}")
    
    try:
        # Create a new backup
        backup_info = create_backup()
        print(f"Backup created: {{backup_info['backup_path']}}")
        
        # Clean up old backups
        cleanup_old_backups()
        
        print(f"Scheduled backup completed successfully")
        return 0
    except Exception as e:
        print(f"Scheduled backup failed: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
            
            # Write the script to file
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            # Create a README with scheduling instructions
            readme_path = self.backup_dir / "BACKUP_README.md"
            
            readme_content = f"""# Database Backup System

## Overview
This directory contains database backups and backup management scripts for the Simulation Validation Framework.

## Scheduled Backups
A backup script has been generated to facilitate scheduled backups:

- **Script:** {script_path}
- **Backup Interval:** {interval_hours} hours
- **Retention:** Keep the {keep_backups} most recent backups
- **Compression:** {"Enabled" if compress else "Disabled"}

## Setting Up Scheduled Backups

### Using cron (Linux/macOS)
To set up automatic backups every {interval_hours} hours:

1. Open your crontab:
   ```
   crontab -e
   ```

2. Add the following line:
   ```
   0 */{interval_hours} * * * /usr/bin/python3 {script_path}
   ```

3. Save and exit.

### Using Task Scheduler (Windows)
1. Open Task Scheduler
2. Create a new task to run the script every {interval_hours} hours
3. Action: Start a program
4. Program: python.exe
5. Arguments: "{script_path}"

## Manual Backup
You can run a manual backup at any time:
```
python3 {script_path}
```

## Restore Process
To restore from a backup, use the SimulationValidationDBIntegration class with the DatabaseBackupManager:

```python
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
from duckdb_api.simulation_validation.db_performance_optimization import DatabaseBackupManager

# Initialize the database integration
db_integration = SimulationValidationDBIntegration(db_path="./benchmark_db.duckdb")

# Initialize the backup manager
backup_manager = DatabaseBackupManager(db_integration, backup_dir="{self.backup_dir}")

# List available backups
backups = backup_manager.list_backups()
for backup in backups:
    print(f"Backup: {{backup['backup_name']}}, Date: {{backup['timestamp']}}")

# Restore a backup (replace with actual backup path)
result = backup_manager.restore_backup(
    backup_path="{self.backup_dir}/backup_example.duckdb.gz",
    compressed=True
)
```

Last updated: {datetime.datetime.now().isoformat()}
"""
            
            # Write the README to file
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            return {
                "success": True,
                "script_path": str(script_path),
                "readme_path": str(readme_path),
                "interval_hours": interval_hours,
                "keep_backups": keep_backups,
                "compress": compress,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error setting up scheduled backups: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }


class OptimizedSimulationValidationDBIntegration(SimulationValidationDBIntegration):
    """
    Optimized version of the SimulationValidationDBIntegration class with performance enhancements.
    
    This class extends the base DB integration class with:
    - Query caching
    - Batch operations
    - Query optimization
    - Database maintenance utilities
    - Backup and restore functionality
    """
    
    def __init__(
        self, 
        db_path: str = "./benchmark_db.duckdb", 
        debug: bool = False,
        enable_caching: bool = True,
        cache_ttl: int = 300,
        batch_size: int = 100,
        backup_dir: str = "./backups",
        auto_optimize: bool = True
    ):
        """
        Initialize the optimized database integration.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
            enable_caching: Whether to enable query caching
            cache_ttl: Time-to-live for cache entries in seconds
            batch_size: Size of batches for batch operations
            backup_dir: Directory to store backups
            auto_optimize: Whether to automatically apply optimizations
        """
        # Initialize the base class
        super().__init__(db_path=db_path, debug=debug)
        
        # Initialize performance optimization components
        self.query_cache = QueryCache(ttl=cache_ttl, max_size=100) if enable_caching else None
        self.batch_operations = BatchOperations(self, batch_size=batch_size)
        self.query_optimizer = QueryOptimizer(self)
        self.maintenance = DatabaseMaintenance(self)
        self.backup_manager = DatabaseBackupManager(self, backup_dir=backup_dir)
        
        self.enable_caching = enable_caching
        self.auto_optimize = auto_optimize
        
        # Apply initial optimizations if auto-optimize is enabled
        if self.auto_optimize:
            try:
                self.query_optimizer.create_indexes()
                self.query_optimizer.analyze_tables()
                logger.info("Applied automatic database optimizations")
            except Exception as e:
                logger.warning(f"Error applying automatic optimizations: {e}")
    
    def _execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Tuple]:
        """
        Execute a query with optimizations.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            use_cache: Whether to use query caching
            
        Returns:
            Query results
        """
        # Check if caching is enabled and requested
        if self.enable_caching and use_cache and not query.upper().startswith(("INSERT", "UPDATE", "DELETE")):
            # Try to get from cache
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                return cached_result
        
        # Optimize the query
        if self.auto_optimize:
            optimized_query = self.query_optimizer.optimize_query(query)
        else:
            optimized_query = query
        
        # Execute the query
        try:
            conn = self.db_api._get_connection()
            start_time = time.time()
            
            if params:
                result = conn.execute(optimized_query, params).fetchall()
            else:
                result = conn.execute(optimized_query).fetchall()
            
            execution_time = time.time() - start_time
            
            # Log query performance
            if execution_time > 1.0:  # Log slow queries
                logger.warning(f"Slow query ({execution_time:.2f}s): {optimized_query[:100]}...")
            else:
                logger.debug(f"Query executed in {execution_time:.4f}s: {optimized_query[:100]}...")
            
            # Cache the result if caching is enabled and requested
            if self.enable_caching and use_cache and not query.upper().startswith(("INSERT", "UPDATE", "DELETE")):
                self.query_cache.set(query, params, result)
            
            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Tuple]:
        """
        Execute a query with optimizations.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            use_cache: Whether to use query caching
            
        Returns:
            Query results
        """
        return self._execute_query(query, params, use_cache)
    
    def invalidate_cache(self, table_name: Optional[str] = None) -> None:
        """
        Invalidate cache entries for a table or all entries.
        
        Args:
            table_name: Name of table to invalidate (or None for all)
        """
        if self.enable_caching and self.query_cache:
            self.query_cache.invalidate(table_name)
    
    def batch_insert_simulation_results(self, results: List[Any]) -> int:
        """
        Insert multiple simulation results in batches.
        
        Args:
            results: List of SimulationResult objects
            
        Returns:
            Number of records inserted
        """
        if not results:
            logger.warning("No simulation results to insert")
            return 0
        
        # Convert SimulationResult objects to database records
        records = []
        for result in results:
            record = self.db_api.schema.simulation_result_to_db_dict(result)
            records.append(record)
        
        # Use batch operations to insert
        count = self.batch_operations.batch_insert(
            table_name="simulation_results",
            records=records
        )
        
        # Invalidate cache
        if self.enable_caching and self.query_cache:
            self.query_cache.invalidate("simulation_results")
        
        return count
    
    def batch_insert_hardware_results(self, results: List[Any]) -> int:
        """
        Insert multiple hardware results in batches.
        
        Args:
            results: List of HardwareResult objects
            
        Returns:
            Number of records inserted
        """
        if not results:
            logger.warning("No hardware results to insert")
            return 0
        
        # Convert HardwareResult objects to database records
        records = []
        for result in results:
            record = self.db_api.schema.hardware_result_to_db_dict(result)
            records.append(record)
        
        # Use batch operations to insert
        count = self.batch_operations.batch_insert(
            table_name="hardware_results",
            records=records
        )
        
        # Invalidate cache
        if self.enable_caching and self.query_cache:
            self.query_cache.invalidate("hardware_results")
        
        return count
    
    def batch_insert_validation_results(self, results: List[Any]) -> int:
        """
        Insert multiple validation results in batches.
        
        Args:
            results: List of ValidationResult objects
            
        Returns:
            Number of records inserted
        """
        if not results:
            logger.warning("No validation results to insert")
            return 0
        
        # First, store the simulation and hardware results
        sim_results = [result.simulation_result for result in results]
        hw_results = [result.hardware_result for result in results]
        
        self.batch_insert_simulation_results(sim_results)
        self.batch_insert_hardware_results(hw_results)
        
        # Now, prepare validation records
        records = []
        for result in results:
            # Find the IDs of the stored simulation and hardware results
            sim_id = self._get_simulation_result_id(result.simulation_result)
            hw_id = self._get_hardware_result_id(result.hardware_result)
            
            if not sim_id or not hw_id:
                logger.error("Failed to find simulation or hardware result IDs")
                continue
            
            record = self.db_api.schema.validation_result_to_db_dict(
                result, sim_id, hw_id
            )
            records.append(record)
        
        # Use batch operations to insert
        count = self.batch_operations.batch_insert(
            table_name="validation_results",
            records=records
        )
        
        # Invalidate cache
        if self.enable_caching and self.query_cache:
            self.query_cache.invalidate("validation_results")
        
        return count
    
    def _get_simulation_result_id(self, result: Any) -> Optional[str]:
        """Get the ID of a stored simulation result."""
        try:
            # Look up by unique identifiers
            query = """
            SELECT id FROM simulation_results
            WHERE model_id = ? AND hardware_id = ? AND timestamp = ?
            LIMIT 1
            """
            params = [result.model_id, result.hardware_id, result.timestamp]
            
            query_result = self._execute_query(query, params, use_cache=False)
            if query_result and len(query_result) > 0:
                return query_result[0][0]
            return None
        except Exception as e:
            logger.error(f"Error getting simulation result ID: {e}")
            return None
    
    def _get_hardware_result_id(self, result: Any) -> Optional[str]:
        """Get the ID of a stored hardware result."""
        try:
            # Look up by unique identifiers
            query = """
            SELECT id FROM hardware_results
            WHERE model_id = ? AND hardware_id = ? AND timestamp = ?
            LIMIT 1
            """
            params = [result.model_id, result.hardware_id, result.timestamp]
            
            query_result = self._execute_query(query, params, use_cache=False)
            if query_result and len(query_result) > 0:
                return query_result[0][0]
            return None
        except Exception as e:
            logger.error(f"Error getting hardware result ID: {e}")
            return None
    
    def optimize_database(self) -> Dict[str, Any]:
        """
        Apply all database optimizations.
        
        Returns:
            Dictionary with optimization results
        """
        results = {}
        
        try:
            # Create indexes
            start_time = time.time()
            self.query_optimizer.create_indexes()
            results["create_indexes"] = {
                "success": True,
                "time": time.time() - start_time
            }
        except Exception as e:
            results["create_indexes"] = {
                "success": False,
                "error": str(e)
            }
        
        try:
            # Analyze tables
            start_time = time.time()
            self.query_optimizer.analyze_tables()
            results["analyze_tables"] = {
                "success": True,
                "time": time.time() - start_time
            }
        except Exception as e:
            results["analyze_tables"] = {
                "success": False,
                "error": str(e)
            }
        
        try:
            # Vacuum database
            start_time = time.time()
            self.maintenance.vacuum_database()
            results["vacuum_database"] = {
                "success": True,
                "time": time.time() - start_time
            }
        except Exception as e:
            results["vacuum_database"] = {
                "success": False,
                "error": str(e)
            }
        
        try:
            # Check integrity
            start_time = time.time()
            integrity = self.maintenance.check_integrity()
            results["check_integrity"] = {
                "success": integrity["overall"],
                "details": integrity,
                "time": time.time() - start_time
            }
        except Exception as e:
            results["check_integrity"] = {
                "success": False,
                "error": str(e)
            }
        
        # Add overall result
        results["overall"] = all(r.get("success", False) for r in results.values())
        results["timestamp"] = datetime.datetime.now().isoformat()
        
        return results
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get database performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {}
        
        # Get database size
        stats["database_size_bytes"] = self.maintenance.get_database_size()
        stats["database_size_mb"] = stats["database_size_bytes"] / (1024 * 1024)
        
        # Get table statistics
        stats["tables"] = self.query_optimizer.get_table_statistics()
        
        # Get query cache statistics if enabled
        if self.enable_caching and self.query_cache:
            stats["query_cache"] = {
                "enabled": True,
                "ttl": self.query_cache.ttl,
                "max_size": self.query_cache.max_size,
                "current_size": len(self.query_cache.cache),
                "cache_hit_ratio": 0.0  # Would need to track hits/misses to calculate
            }
        else:
            stats["query_cache"] = {"enabled": False}
        
        # Get batch operation statistics
        stats["batch_operations"] = {
            "batch_size": self.batch_operations.batch_size
        }
        
        # Include timestamp
        stats["timestamp"] = datetime.datetime.now().isoformat()
        
        return stats
    
    def run_maintenance(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run database maintenance tasks.
        
        Args:
            tasks: List of maintenance tasks to run (default: all)
            
        Returns:
            Dictionary with maintenance results
        """
        if tasks is None:
            tasks = ["vacuum", "integrity_check", "cleanup_old_data", "rebuild_indexes"]
        
        results = {}
        
        for task in tasks:
            try:
                if task == "vacuum":
                    start_time = time.time()
                    self.maintenance.vacuum_database()
                    results["vacuum"] = {
                        "success": True,
                        "time": time.time() - start_time
                    }
                
                elif task == "integrity_check":
                    start_time = time.time()
                    integrity = self.maintenance.check_integrity()
                    results["integrity_check"] = {
                        "success": integrity["overall"],
                        "details": integrity,
                        "time": time.time() - start_time
                    }
                
                elif task == "cleanup_old_data":
                    start_time = time.time()
                    cleanup = self.maintenance.cleanup_old_data(retention_days=365)
                    results["cleanup_old_data"] = {
                        "success": "error" not in cleanup,
                        "details": cleanup,
                        "time": time.time() - start_time
                    }
                
                elif task == "rebuild_indexes":
                    start_time = time.time()
                    self.maintenance.rebuild_indexes()
                    results["rebuild_indexes"] = {
                        "success": True,
                        "time": time.time() - start_time
                    }
                
                elif task == "monitor_performance":
                    start_time = time.time()
                    monitoring = self.maintenance.monitor_database_performance()
                    results["monitor_performance"] = {
                        "success": "error" not in monitoring,
                        "details": monitoring,
                        "time": time.time() - start_time
                    }
                
                else:
                    results[task] = {
                        "success": False,
                        "error": f"Unknown maintenance task: {task}"
                    }
            
            except Exception as e:
                results[task] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Add overall result
        results["overall"] = all(r.get("success", False) for r in results.values())
        results["timestamp"] = datetime.datetime.now().isoformat()
        
        # Clear the cache after maintenance
        if self.enable_caching and self.query_cache:
            self.query_cache.clear()
        
        return results
    
    def backup_database(
        self, 
        backup_name: Optional[str] = None,
        compress: bool = True,
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Create a database backup.
        
        Args:
            backup_name: Custom name for the backup
            compress: Whether to compress the backup
            verify: Whether to verify the backup after creation
            
        Returns:
            Dictionary with backup information
        """
        return self.backup_manager.create_backup(
            backup_name=backup_name,
            compress=compress,
            verify=verify
        )
    
    def restore_database(
        self, 
        backup_path: str,
        target_path: Optional[str] = None,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Restore a database from backup.
        
        Args:
            backup_path: Path to the backup file
            target_path: Path to restore to
            compressed: Whether the backup is compressed
            
        Returns:
            Dictionary with restoration results
        """
        return self.backup_manager.restore_backup(
            backup_path=backup_path,
            target_path=target_path,
            compressed=compressed
        )
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        return self.backup_manager.list_backups()
    
    def close(self):
        """Close the database connection and release resources."""
        super().close()
        
        # Clear cache if enabled
        if self.enable_caching and self.query_cache:
            self.query_cache.clear()
        
        logger.info("Closed database connection and released resources")


# Utility function to get an instance of the optimized DB integration
def get_optimized_db_integration(
    db_path: str = "./benchmark_db.duckdb",
    debug: bool = False,
    enable_caching: bool = True,
    cache_ttl: int = 300,
    batch_size: int = 100,
    backup_dir: str = "./backups",
    auto_optimize: bool = True
) -> OptimizedSimulationValidationDBIntegration:
    """
    Get an instance of the optimized database integration.
    
    Args:
        db_path: Path to the DuckDB database
        debug: Enable debug logging
        enable_caching: Whether to enable query caching
        cache_ttl: Time-to-live for cache entries in seconds
        batch_size: Size of batches for batch operations
        backup_dir: Directory to store backups
        auto_optimize: Whether to automatically apply optimizations
        
    Returns:
        OptimizedSimulationValidationDBIntegration instance
    """
    return OptimizedSimulationValidationDBIntegration(
        db_path=db_path,
        debug=debug,
        enable_caching=enable_caching,
        cache_ttl=cache_ttl,
        batch_size=batch_size,
        backup_dir=backup_dir,
        auto_optimize=auto_optimize
    )