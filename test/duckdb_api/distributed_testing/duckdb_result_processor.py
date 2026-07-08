#!/usr/bin/env python3
"""
Distributed Testing Framework - DuckDB Result Processor

This module implements the integration between worker nodes and the DuckDB database
for efficient test result storage and retrieval. It provides transaction support,
connection pooling, and batch operations to optimize database interactions.

Core features:
- Connection pooling for high-concurrency environments
- Transactional result storage with rollback capability
- Batch result insertion for efficiency
- Result validation to ensure data integrity
- Consistent error handling and logging
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("duckdb_result_processor")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.error("DuckDB not available. Result processor cannot function.")
    DUCKDB_AVAILABLE = False

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class DuckDBResultProcessor:
    """Processes test results and stores them directly in DuckDB."""
    
    def __init__(self, db_path, pool_size=5):
        """Initialize with database path and connection pool size.
        
        Args:
            db_path: Path to the DuckDB database file
            pool_size: Number of database connections to maintain in the pool
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required for DuckDBResultProcessor")
            
        self.db_path = db_path
        self.pool_size = pool_size
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.result_schema = self._get_result_schema()
        self.initialize_pool()
        
        # Create schema if needed
        self._ensure_schema_exists()
    
    def initialize_pool(self):
        """Create connection pool for database access."""
        with self.pool_lock:
            for _ in range(self.pool_size):
                conn = duckdb.connect(self.db_path)
                self.connection_pool.append(conn)
    
    def get_connection(self):
        """Get a connection from the pool."""
        with self.pool_lock:
            if not self.connection_pool:
                return duckdb.connect(self.db_path)
            return self.connection_pool.pop()
    
    def release_connection(self, conn):
        """Return a connection to the pool."""
        with self.pool_lock:
            self.connection_pool.append(conn)
    
    def store_result(self, result_data):
        """Store a single test result in the database.
        
        Args:
            result_data: Dictionary containing test result data
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        try:
            # Validate result data
            self._validate_result(result_data)
            
            # Convert to appropriate format
            db_record = self._convert_to_db_format(result_data)
            
            # Store in database with transaction support
            conn.execute("BEGIN TRANSACTION")
            conn.execute("""
                INSERT INTO test_results (
                    test_id, worker_id, model_name, hardware_type, 
                    execution_time, success, error_message, timestamp,
                    memory_usage, details, power_consumption, test_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                db_record['test_id'], db_record['worker_id'],
                db_record['model_name'], db_record['hardware_type'],
                db_record['execution_time'], db_record['success'],
                db_record['error_message'], db_record['timestamp'],
                db_record['memory_usage'], json.dumps(db_record['details']),
                db_record['power_consumption'], db_record['test_type']
            ])
            conn.execute("COMMIT")
            logger.debug(f"Stored result {db_record['test_id']} in database")
            return True
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Error storing result: {e}")
            return False
        finally:
            self.release_connection(conn)
    
    def store_batch_results(self, results):
        """Store multiple test results efficiently.
        
        Args:
            results: List of result dictionaries to store
            
        Returns:
            Tuple[bool, List]: (success, list of failed results)
        """
        conn = self.get_connection()
        failed_results = []
        
        try:
            conn.execute("BEGIN TRANSACTION")
            
            for result in results:
                try:
                    self._validate_result(result)
                    db_record = self._convert_to_db_format(result)
                    
                    conn.execute("""
                        INSERT INTO test_results (
                            test_id, worker_id, model_name, hardware_type, 
                            execution_time, success, error_message, timestamp,
                            memory_usage, details, power_consumption, test_type
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        db_record['test_id'], db_record['worker_id'],
                        db_record['model_name'], db_record['hardware_type'],
                        db_record['execution_time'], db_record['success'],
                        db_record['error_message'], db_record['timestamp'],
                        db_record['memory_usage'], json.dumps(db_record['details']),
                        db_record['power_consumption'], db_record['test_type']
                    ])
                except Exception as e:
                    logger.warning(f"Error processing result {result.get('test_id', 'unknown')}: {e}")
                    failed_results.append(result)
            
            conn.execute("COMMIT")
            logger.info(f"Stored {len(results) - len(failed_results)} of {len(results)} results in database")
            success = len(failed_results) == 0
            return success, failed_results
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Error storing batch results: {e}")
            return False, results
        finally:
            self.release_connection(conn)
    
    def _validate_result(self, result):
        """Validate result data structure and types.
        
        Args:
            result: Result data to validate
            
        Raises:
            ValueError: If the result data is invalid
        """
        required_fields = ['test_id', 'worker_id', 'model_name', 'hardware_type', 'success']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
                
        # Type validation
        if not isinstance(result.get('success'), bool):
            raise ValueError("Field 'success' must be a boolean")
        
        # Value validation
        if 'execution_time' in result and result['execution_time'] is not None:
            if not isinstance(result['execution_time'], (int, float)) or result['execution_time'] < 0:
                raise ValueError("Field 'execution_time' must be a non-negative number")
                
        if 'memory_usage' in result and result['memory_usage'] is not None:
            if not isinstance(result['memory_usage'], (int, float)) or result['memory_usage'] < 0:
                raise ValueError("Field 'memory_usage' must be a non-negative number")
    
    def _convert_to_db_format(self, result):
        """Convert result to database-compatible format.
        
        Args:
            result: Result data to convert
            
        Returns:
            Dict: Database-compatible record
        """
        db_record = {
            'test_id': result.get('test_id'),
            'worker_id': result.get('worker_id'),
            'model_name': result.get('model_name'),
            'hardware_type': result.get('hardware_type'),
            'execution_time': result.get('execution_time', 0.0),
            'success': result.get('success', False),
            'error_message': result.get('error_message', ''),
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'memory_usage': result.get('memory_usage', 0.0),
            'details': result.get('details', {}),
            'power_consumption': result.get('power_consumption', 0.0),
            'test_type': result.get('test_type', 'unknown')
        }
        
        # Ensure details is JSON serializable
        if isinstance(db_record['details'], str):
            try:
                db_record['details'] = json.loads(db_record['details'])
            except json.JSONDecodeError:
                db_record['details'] = {'raw_details': db_record['details']}
        
        return db_record
    
    def _ensure_schema_exists(self):
        """Create database schema if it doesn't exist."""
        conn = self.get_connection()
        try:
            # Check if test_results table exists
            table_exists = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='test_results'
            """).fetchone() is not None
            
            if not table_exists:
                logger.info("Creating test_results table")
                
                conn.execute("""
                    CREATE TABLE test_results (
                        id INTEGER PRIMARY KEY,
                        test_id VARCHAR UNIQUE,
                        worker_id VARCHAR,
                        model_name VARCHAR,
                        hardware_type VARCHAR,
                        execution_time FLOAT,
                        success BOOLEAN,
                        error_message VARCHAR,
                        timestamp VARCHAR,
                        memory_usage FLOAT,
                        details JSON,
                        power_consumption FLOAT,
                        test_type VARCHAR
                    )
                """)
                
                # Create indices for faster querying
                conn.execute("CREATE INDEX idx_test_results_model_name ON test_results(model_name)")
                conn.execute("CREATE INDEX idx_test_results_hardware_type ON test_results(hardware_type)")
                conn.execute("CREATE INDEX idx_test_results_worker_id ON test_results(worker_id)")
                conn.execute("CREATE INDEX idx_test_results_timestamp ON test_results(timestamp)")
                conn.execute("CREATE INDEX idx_test_results_success ON test_results(success)")
                
                logger.info("Created test_results table and indices")
                
            # Check if worker_metrics table exists
            worker_metrics_exists = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='worker_metrics'
            """).fetchone() is not None
            
            if not worker_metrics_exists:
                logger.info("Creating worker_metrics table")
                
                conn.execute("""
                    CREATE TABLE worker_metrics (
                        id INTEGER PRIMARY KEY,
                        worker_id VARCHAR,
                        timestamp VARCHAR,
                        cpu_usage FLOAT,
                        memory_usage FLOAT,
                        gpu_usage FLOAT,
                        gpu_memory_usage FLOAT,
                        temperature FLOAT,
                        power_usage FLOAT,
                        active_tasks INTEGER,
                        details JSON
                    )
                """)
                
                conn.execute("CREATE INDEX idx_worker_metrics_worker_id ON worker_metrics(worker_id)")
                conn.execute("CREATE INDEX idx_worker_metrics_timestamp ON worker_metrics(timestamp)")
                
                logger.info("Created worker_metrics table and indices")
        except Exception as e:
            logger.error(f"Error ensuring schema exists: {e}")
            raise
        finally:
            self.release_connection(conn)
    
    def _get_result_schema(self):
        """Get the schema for test results."""
        return {
            'test_id': str,
            'worker_id': str,
            'model_name': str,
            'hardware_type': str,
            'execution_time': float,
            'success': bool,
            'error_message': str,
            'timestamp': str,
            'memory_usage': float,
            'details': dict,
            'power_consumption': float,
            'test_type': str
        }
    
    def query_results(self, query, params=None):
        """Run a custom query on the results database.
        
        Args:
            query: SQL query to execute
            params: Parameters for the query
            
        Returns:
            List: Query results
        """
        conn = self.get_connection()
        try:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
        finally:
            self.release_connection(conn)
    
    def get_results_by_model(self, model_name, limit=100):
        """Get test results for a specific model.
        
        Args:
            model_name: Name of the model to query
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        return self.query_results(
            "SELECT * FROM test_results WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
            [model_name, limit]
        )
    
    def get_results_by_hardware(self, hardware_type, limit=100):
        """Get test results for a specific hardware type.
        
        Args:
            hardware_type: Type of hardware to query
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        return self.query_results(
            "SELECT * FROM test_results WHERE hardware_type = ? ORDER BY timestamp DESC LIMIT ?",
            [hardware_type, limit]
        )
    
    def get_results_by_worker(self, worker_id, limit=100):
        """Get test results for a specific worker.
        
        Args:
            worker_id: ID of the worker to query
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        return self.query_results(
            "SELECT * FROM test_results WHERE worker_id = ? ORDER BY timestamp DESC LIMIT ?",
            [worker_id, limit]
        )
    
    def get_results_by_timerange(self, start_time, end_time=None, limit=100):
        """Get test results within a time range.
        
        Args:
            start_time: Start of the time range (ISO format)
            end_time: End of the time range (ISO format, defaults to now)
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        if end_time is None:
            end_time = datetime.now().isoformat()
            
        return self.query_results(
            "SELECT * FROM test_results WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT ?",
            [start_time, end_time, limit]
        )
    
    def get_result_by_id(self, test_id):
        """Get a specific test result by ID.
        
        Args:
            test_id: ID of the test to retrieve
            
        Returns:
            Dict or None: Test result or None if not found
        """
        results = self.query_results(
            "SELECT * FROM test_results WHERE test_id = ?",
            [test_id]
        )
        
        if results:
            return results[0]
        return None
    
    def get_summary_by_model_hardware(self):
        """Get a summary of test results grouped by model and hardware type.
        
        Returns:
            List: Summary of test results
        """
        return self.query_results("""
            SELECT 
                model_name, 
                hardware_type, 
                COUNT(*) as test_count,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                AVG(execution_time) as avg_execution_time,
                AVG(memory_usage) as avg_memory_usage,
                AVG(power_consumption) as avg_power_consumption
            FROM test_results
            GROUP BY model_name, hardware_type
            ORDER BY model_name, hardware_type
        """)
    
    def store_worker_metrics(self, worker_id, metrics):
        """Store worker metrics in the database.
        
        Args:
            worker_id: ID of the worker
            metrics: Dictionary of worker metrics
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        try:
            timestamp = metrics.get('timestamp', datetime.now().isoformat())
            cpu_usage = metrics.get('cpu_usage', 0.0)
            memory_usage = metrics.get('memory_usage', 0.0)
            gpu_usage = metrics.get('gpu_usage', 0.0)
            gpu_memory_usage = metrics.get('gpu_memory_usage', 0.0)
            temperature = metrics.get('temperature', 0.0)
            power_usage = metrics.get('power_usage', 0.0)
            active_tasks = metrics.get('active_tasks', 0)
            details = metrics.get('details', {})
            
            conn.execute("""
                INSERT INTO worker_metrics (
                    worker_id, timestamp, cpu_usage, memory_usage, gpu_usage,
                    gpu_memory_usage, temperature, power_usage, active_tasks, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                worker_id, timestamp, cpu_usage, memory_usage, gpu_usage,
                gpu_memory_usage, temperature, power_usage, active_tasks,
                json.dumps(details)
            ])
            
            return True
        except Exception as e:
            logger.error(f"Error storing worker metrics: {e}")
            return False
        finally:
            self.release_connection(conn)
    
    def get_worker_metrics(self, worker_id, limit=100):
        """Get metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker to query
            limit: Maximum number of metrics to return
            
        Returns:
            List: Worker metrics
        """
        return self.query_results(
            "SELECT * FROM worker_metrics WHERE worker_id = ? ORDER BY timestamp DESC LIMIT ?",
            [worker_id, limit]
        )
    
    def close(self):
        """Close all database connections."""
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool = []
            logger.info(f"Closed {self.pool_size} database connections")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DuckDB Result Processor")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    parser.add_argument("--test", action="store_true", help="Run a test")
    args = parser.parse_args()
    
    processor = DuckDBResultProcessor(args.db_path)
    
    if args.test:
        logger.info("Running test")
        
        # Store a single test result
        test_result = {
            "test_id": str(uuid.uuid4()),
            "worker_id": "test-worker",
            "model_name": "test-model",
            "hardware_type": "cpu",
            "execution_time": 10.5,
            "success": True,
            "memory_usage": 256.0,
            "test_type": "unit-test"
        }
        
        result = processor.store_result(test_result)
        logger.info(f"Store result: {result}")
        
        # Store batch results
        batch_results = []
        for i in range(5):
            batch_results.append({
                "test_id": str(uuid.uuid4()),
                "worker_id": "test-worker",
                "model_name": f"test-model-{i}",
                "hardware_type": "cpu",
                "execution_time": 10.5 + i,
                "success": i % 2 == 0,
                "memory_usage": 256.0 + i * 10,
                "test_type": "batch-test"
            })
        
        batch_result, failed = processor.store_batch_results(batch_results)
        logger.info(f"Store batch results: {batch_result}, failed: {len(failed)}")
        
        # Query results
        results = processor.get_results_by_model("test-model")
        logger.info(f"Results by model: {len(results)}")
        
        # Get summary
        summary = processor.get_summary_by_model_hardware()
        logger.info(f"Summary: {summary}")
        
        processor.close()