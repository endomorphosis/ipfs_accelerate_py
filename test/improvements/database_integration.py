#!/usr/bin/env python3
"""
Database Integration Module

This module provides standardized database integration for all test generators,
benchmark runners, and test execution frameworks. It handles:

1. Consistent database connections and schema management
2. Standardized result storage patterns
3. Proper error handling and transaction management
4. Migration utilities from JSON to DuckDB
5. Test run tracking and management

Usage:
  from improvements.database_integration import get_db_connection, store_test_result
"""

import os
import sys
import json
import time
import logging
import datetime
import tempfile
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for DuckDB availability
try:
    import duckdb
    import pandas as pd
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB or pandas not available, database functionality will be limited")

# Environment variables
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")
BENCHMARK_DB_PATH = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")

# Database connection cache
_DB_CONNECTIONS = {}

def get_db_connection(db_path: Optional[str] = None, read_only: bool = False) -> Optional['duckdb.DuckDBPyConnection']:
    """
    Get a database connection with proper caching and consistent configuration.
    
    Args:
        db_path: Path to the database file. Defaults to BENCHMARK_DB_PATH env var.
        read_only: Whether to open the connection in read-only mode.
        
    Returns:
        DuckDB connection or None if DuckDB is not available
    """
    if not DUCKDB_AVAILABLE:
        logger.warning("DuckDB not available, returning None")
        return None
    
    # Use the provided path or the environment variable
    db_path = db_path or BENCHMARK_DB_PATH
    
    # Create a cache key that accounts for the path and access mode
    cache_key = f"{db_path}:{read_only}"
    
    # Check if we already have a connection for this path and mode
    if cache_key in _DB_CONNECTIONS:
        # Check if the connection is still valid
        try:
            conn = _DB_CONNECTIONS[cache_key]
            # Execute a simple query to check if connection is still valid
            conn.execute("SELECT 1").fetchone()
            return conn
        except Exception:
            # Connection is invalid, remove it from cache
            del _DB_CONNECTIONS[cache_key]
    
    try:
        # Create the directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Open the connection
        conn = duckdb.connect(db_path, read_only=read_only)
        
        # Cache the connection for reuse
        _DB_CONNECTIONS[cache_key] = conn
        
        # Initialize the schema if this is a new database
        if not read_only:
            _ensure_schema(conn)
        
        return conn
    except Exception as e:
        logger.error(f"Error opening database connection to {db_path}: {e}")
        return None

def close_all_connections():
    """Close all open database connections."""
    for key, conn in list(_DB_CONNECTIONS.items()):
        try:
            conn.close()
            del _DB_CONNECTIONS[key]
        except Exception as e:
            logger.error(f"Error closing connection {key}: {e}")

def _ensure_schema(conn):
    """
    Ensure the database has the required schema.
    
    Args:
        conn: Database connection to use
    """
    # Check if the models table exists as a proxy for schema initialization
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
    ).fetchone() is not None
    
    if not table_exists:
        # Create minimal schema
        conn.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            run_id INTEGER PRIMARY KEY,
            test_name VARCHAR NOT NULL,
            test_type VARCHAR NOT NULL,
            started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            metadata JSON
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_family VARCHAR,
            model_type VARCHAR,
            task VARCHAR,
            metadata JSON,
            UNIQUE(model_name)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_platforms (
            hardware_id INTEGER PRIMARY KEY,
            hardware_type VARCHAR NOT NULL,
            hardware_name VARCHAR,
            device_count INTEGER,
            version VARCHAR,
            metadata JSON,
            UNIQUE(hardware_type)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_results (
            result_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER,
            hardware_id INTEGER,
            batch_size INTEGER,
            sequence_length INTEGER,
            input_shape VARCHAR,
            throughput_items_per_second FLOAT,
            latency_ms FLOAT,
            memory_mb FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_compatibility (
            compatibility_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER,
            hardware_id INTEGER,
            compatibility_type VARCHAR NOT NULL, -- REAL, SIMULATION, INCOMPATIBLE
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON,
            UNIQUE(model_id, hardware_id),
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            test_result_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            test_name VARCHAR NOT NULL,
            status VARCHAR NOT NULL, -- PASS, FAIL, ERROR
            execution_time_seconds FLOAT,
            model_id INTEGER,
            hardware_id INTEGER,
            error_message VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS web_platform_results (
            result_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER,
            browser VARCHAR,
            browser_version VARCHAR,
            platform VARCHAR, -- webnn, webgpu
            optimization_flags JSON,
            initialization_time_ms FLOAT,
            first_inference_time_ms FLOAT,
            subsequent_inference_time_ms FLOAT,
            memory_mb FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS integration_test_results (
            test_result_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            test_module VARCHAR NOT NULL,
            test_class VARCHAR,
            test_name VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            execution_time_seconds FLOAT,
            hardware_id INTEGER,
            model_id INTEGER,
            error_message VARCHAR,
            error_traceback VARCHAR,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_implementations (
            implementation_id INTEGER PRIMARY KEY,
            model_type VARCHAR NOT NULL,
            file_path VARCHAR NOT NULL,
            generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_category VARCHAR,
            hardware_support JSON,
            primary_task VARCHAR,
            cross_platform BOOLEAN DEFAULT FALSE,
            UNIQUE(model_type)
        )
        """)
        
        logger.info("Database schema initialized successfully")

def create_test_run(test_name: str, test_type: str, metadata: Dict = None) -> Optional[int]:
    """
    Create a new test run entry in the database.
    
    Args:
        test_name: Name of the test
        test_type: Type of test (performance, hardware_compatibility, web_platform, etc.)
        metadata: Additional metadata for the test run
        
    Returns:
        run_id of the created test run, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Insert test run
        metadata_json = json.dumps(metadata) if metadata else None
        conn.execute(
            """
            INSERT INTO test_runs (test_name, test_type, metadata)
            VALUES (?, ?, ?)
            """,
            [test_name, test_type, metadata_json]
        )
        
        # Get the run_id
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.info(f"Created test run: {test_name} (ID: {run_id})")
        return run_id
    except Exception as e:
        logger.error(f"Error creating test run: {e}")
        return None

def get_or_create_test_run(test_name: str, test_type: str, metadata: Dict = None) -> Optional[int]:
    """
    Get an existing test run ID or create a new one.
    
    Args:
        test_name: Name of the test
        test_type: Type of test
        metadata: Test metadata
        
    Returns:
        run_id: ID of the test run, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Look for an existing active test run
        result = conn.execute(
            """
            SELECT run_id FROM test_runs 
            WHERE test_name = ? AND completed_at IS NULL 
            ORDER BY started_at DESC LIMIT 1
            """,
            [test_name]
        ).fetchone()
        
        if result:
            run_id = result[0]
            logger.debug(f"Found active test run: {test_name} (ID: {run_id})")
            return run_id
        
        # Create a new test run
        return create_test_run(test_name, test_type, metadata)
    except Exception as e:
        logger.error(f"Error getting or creating test run: {e}")
        return None

def complete_test_run(run_id: int) -> bool:
    """
    Mark a test run as completed.
    
    Args:
        run_id: ID of the test run to complete
        
    Returns:
        True if successful, False otherwise
    """
    if not DUCKDB_AVAILABLE:
        return False
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        conn.execute(
            """
            UPDATE test_runs 
            SET completed_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            [run_id]
        )
        logger.info(f"Completed test run ID: {run_id}")
        return True
    except Exception as e:
        logger.error(f"Error completing test run {run_id}: {e}")
        return False

def get_or_create_model(model_name: str, model_family: str = None, model_type: str = None, 
                        task: str = None, metadata: Dict = None) -> Optional[int]:
    """
    Get or create a model entry in the database.
    
    Args:
        model_name: Name of the model
        model_family: Model family (bert, t5, etc.)
        model_type: Type of model (text, vision, etc.)
        task: Primary task of the model
        metadata: Additional metadata
        
    Returns:
        model_id: ID of the model, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Check if model exists
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?",
            [model_name]
        ).fetchone()
        
        if result:
            model_id = result[0]
            
            # Update model information if provided
            if model_family or model_type or task or metadata:
                update_fields = []
                update_values = []
                
                if model_family:
                    update_fields.append("model_family = ?")
                    update_values.append(model_family)
                
                if model_type:
                    update_fields.append("model_type = ?")
                    update_values.append(model_type)
                
                if task:
                    update_fields.append("task = ?")
                    update_values.append(task)
                
                if metadata:
                    update_fields.append("metadata = ?")
                    update_values.append(json.dumps(metadata))
                
                if update_fields:
                    conn.execute(
                        f"UPDATE models SET {', '.join(update_fields)} WHERE model_id = ?",
                        update_values + [model_id]
                    )
            
            return model_id
        
        # Create new model entry
        metadata_json = json.dumps(metadata) if metadata else None
        conn.execute(
            """
            INSERT INTO models (model_name, model_family, model_type, task, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            [model_name, model_family, model_type, task, metadata_json]
        )
        
        model_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Created model: {model_name} (ID: {model_id})")
        return model_id
    except Exception as e:
        logger.error(f"Error creating model record for {model_name}: {e}")
        return None

def get_or_create_hardware(hardware_type: str, hardware_name: str = None, 
                          device_count: int = None, version: str = None, 
                          metadata: Dict = None) -> Optional[int]:
    """
    Get or create a hardware platform entry in the database.
    
    Args:
        hardware_type: Type of hardware (cpu, cuda, etc.)
        hardware_name: Name of the hardware
        device_count: Number of devices
        version: Hardware version
        metadata: Additional metadata
        
    Returns:
        hardware_id: ID of the hardware, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Check if hardware exists
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
            [hardware_type]
        ).fetchone()
        
        if result:
            hardware_id = result[0]
            
            # Update hardware information if provided
            if hardware_name or device_count is not None or version or metadata:
                update_fields = []
                update_values = []
                
                if hardware_name:
                    update_fields.append("hardware_name = ?")
                    update_values.append(hardware_name)
                
                if device_count is not None:
                    update_fields.append("device_count = ?")
                    update_values.append(device_count)
                
                if version:
                    update_fields.append("version = ?")
                    update_values.append(version)
                
                if metadata:
                    update_fields.append("metadata = ?")
                    update_values.append(json.dumps(metadata))
                
                if update_fields:
                    conn.execute(
                        f"UPDATE hardware_platforms SET {', '.join(update_fields)} WHERE hardware_id = ?",
                        update_values + [hardware_id]
                    )
            
            return hardware_id
        
        # Create new hardware entry
        metadata_json = json.dumps(metadata) if metadata else None
        conn.execute(
            """
            INSERT INTO hardware_platforms (hardware_type, hardware_name, device_count, version, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            [hardware_type, hardware_name, device_count, version, metadata_json]
        )
        
        hardware_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Created hardware platform: {hardware_type} (ID: {hardware_id})")
        return hardware_id
    except Exception as e:
        logger.error(f"Error creating hardware record for {hardware_type}: {e}")
        return None

def store_performance_result(run_id: int, model_id: int, hardware_id: int, 
                            batch_size: int, throughput: float = None, 
                            latency: float = None, memory: float = None,
                            sequence_length: int = None, input_shape: str = None,
                            metadata: Dict = None) -> Optional[int]:
    """
    Store a performance benchmark result in the database.
    
    Args:
        run_id: ID of the test run
        model_id: ID of the model
        hardware_id: ID of the hardware platform
        batch_size: Batch size used in the benchmark
        throughput: Throughput in items per second
        latency: Latency in milliseconds
        memory: Memory usage in MB
        sequence_length: Sequence length for text models
        input_shape: Input shape string for vision/audio models
        metadata: Additional metadata
        
    Returns:
        result_id: ID of the created result, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        metadata_json = json.dumps(metadata) if metadata else None
        conn.execute(
            """
            INSERT INTO performance_results (
                run_id, model_id, hardware_id, batch_size, 
                sequence_length, input_shape, throughput_items_per_second, 
                latency_ms, memory_mb, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, model_id, hardware_id, batch_size, sequence_length,
             input_shape, throughput, latency, memory, metadata_json]
        )
        
        result_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Stored performance result (ID: {result_id})")
        return result_id
    except Exception as e:
        logger.error(f"Error storing performance result: {e}")
        return None

def store_hardware_compatibility(run_id: int, model_id: int, hardware_id: int,
                                compatibility_type: str, metadata: Dict = None) -> Optional[int]:
    """
    Store a hardware compatibility result in the database.
    
    Args:
        run_id: ID of the test run
        model_id: ID of the model
        hardware_id: ID of the hardware platform
        compatibility_type: Type of compatibility (REAL, SIMULATION, INCOMPATIBLE)
        metadata: Additional metadata
        
    Returns:
        compatibility_id: ID of the created compatibility record, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Check if the record already exists
        result = conn.execute(
            """
            SELECT compatibility_id FROM hardware_compatibility
            WHERE model_id = ? AND hardware_id = ?
            """,
            [model_id, hardware_id]
        ).fetchone()
        
        if result:
            compatibility_id = result[0]
            # Update existing record
            conn.execute(
                """
                UPDATE hardware_compatibility
                SET run_id = ?, compatibility_type = ?, timestamp = CURRENT_TIMESTAMP, metadata = ?
                WHERE compatibility_id = ?
                """,
                [run_id, compatibility_type, metadata_json, compatibility_id]
            )
            logger.debug(f"Updated hardware compatibility (ID: {compatibility_id})")
            return compatibility_id
        
        # Create new record
        conn.execute(
            """
            INSERT INTO hardware_compatibility (
                run_id, model_id, hardware_id, compatibility_type, metadata
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [run_id, model_id, hardware_id, compatibility_type, metadata_json]
        )
        
        compatibility_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Stored hardware compatibility (ID: {compatibility_id})")
        return compatibility_id
    except Exception as e:
        logger.error(f"Error storing hardware compatibility: {e}")
        return None

def store_web_platform_result(run_id: int, model_id: int, browser: str, 
                             platform: str, initialization_time: float = None,
                             first_inference_time: float = None, 
                             subsequent_inference_time: float = None,
                             memory: float = None, browser_version: str = None,
                             optimization_flags: Dict = None, 
                             metadata: Dict = None) -> Optional[int]:
    """
    Store a web platform benchmark result in the database.
    
    Args:
        run_id: ID of the test run
        model_id: ID of the model
        browser: Browser name (chrome, firefox, etc.)
        platform: Web platform (webnn, webgpu)
        initialization_time: Initialization time in milliseconds
        first_inference_time: First inference time in milliseconds
        subsequent_inference_time: Subsequent inference time in milliseconds
        memory: Memory usage in MB
        browser_version: Browser version
        optimization_flags: Optimization flags used
        metadata: Additional metadata
        
    Returns:
        result_id: ID of the created result, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        optimization_flags_json = json.dumps(optimization_flags) if optimization_flags else None
        metadata_json = json.dumps(metadata) if metadata else None
        conn.execute(
            """
            INSERT INTO web_platform_results (
                run_id, model_id, browser, browser_version, platform,
                optimization_flags, initialization_time_ms, first_inference_time_ms,
                subsequent_inference_time_ms, memory_mb, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, model_id, browser, browser_version, platform,
             optimization_flags_json, initialization_time, first_inference_time,
             subsequent_inference_time, memory, metadata_json]
        )
        
        result_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Stored web platform result (ID: {result_id})")
        return result_id
    except Exception as e:
        logger.error(f"Error storing web platform result: {e}")
        return None

def store_test_result(run_id: int, test_name: str, status: str, 
                     execution_time: float = None, model_id: int = None,
                     hardware_id: int = None, error_message: str = None, 
                     metadata: Dict = None) -> Optional[int]:
    """
    Store a test result in the database.
    
    Args:
        run_id: ID of the test run
        test_name: Name of the test
        status: Test status (PASS, FAIL, ERROR)
        execution_time: Test execution time in seconds
        model_id: ID of the model (optional)
        hardware_id: ID of the hardware platform (optional)
        error_message: Error message if test failed
        metadata: Additional metadata
        
    Returns:
        test_result_id: ID of the created test result, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        metadata_json = json.dumps(metadata) if metadata else None
        conn.execute(
            """
            INSERT INTO test_results (
                run_id, test_name, status, execution_time_seconds,
                model_id, hardware_id, error_message, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, test_name, status, execution_time,
             model_id, hardware_id, error_message, metadata_json]
        )
        
        test_result_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Stored test result for {test_name} (ID: {test_result_id})")
        return test_result_id
    except Exception as e:
        logger.error(f"Error storing test result for {test_name}: {e}")
        return None

def store_integration_test_result(run_id: int, test_module: str, test_name: str,
                                 status: str, execution_time: float = None, 
                                 test_class: str = None, model_id: int = None,
                                 hardware_id: int = None, error_message: str = None,
                                 error_traceback: str = None, 
                                 metadata: Dict = None) -> Optional[int]:
    """
    Store an integration test result in the database.
    
    Args:
        run_id: ID of the test run
        test_module: Test module name
        test_name: Test name
        status: Test status (PASS, FAIL, ERROR)
        execution_time: Test execution time in seconds
        test_class: Test class name
        model_id: ID of the model (optional)
        hardware_id: ID of the hardware platform (optional)
        error_message: Error message if test failed
        error_traceback: Error traceback if test failed
        metadata: Additional metadata
        
    Returns:
        test_result_id: ID of the created test result, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        metadata_json = json.dumps(metadata) if metadata else None
        conn.execute(
            """
            INSERT INTO integration_test_results (
                run_id, test_module, test_class, test_name, status,
                execution_time_seconds, hardware_id, model_id,
                error_message, error_traceback, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, test_module, test_class, test_name, status,
             execution_time, hardware_id, model_id,
             error_message, error_traceback, metadata_json]
        )
        
        test_result_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Stored integration test result for {test_module}.{test_name} (ID: {test_result_id})")
        return test_result_id
    except Exception as e:
        logger.error(f"Error storing integration test result for {test_module}.{test_name}: {e}")
        return None

def store_implementation_metadata(model_type: str, file_path: str, 
                                 generation_date: datetime.datetime = None,
                                 model_category: str = None, 
                                 hardware_support: Dict = None,
                                 primary_task: str = None, 
                                 cross_platform: bool = False) -> Optional[int]:
    """
    Store metadata for a generated model implementation.
    
    Args:
        model_type: Type of model (bert, t5, etc.)
        file_path: Path to the implementation file
        generation_date: Generation date
        model_category: Model category
        hardware_support: Hardware support information
        primary_task: Primary task of the model
        cross_platform: Whether the implementation is cross-platform
        
    Returns:
        implementation_id: ID of the created implementation record, or None if creation failed
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        hardware_support_json = json.dumps(hardware_support) if hardware_support else None
        generation_date_str = generation_date.isoformat() if generation_date else datetime.datetime.now().isoformat()
        
        # Check if the record already exists
        result = conn.execute(
            "SELECT implementation_id FROM model_implementations WHERE model_type = ?",
            [model_type]
        ).fetchone()
        
        if result:
            implementation_id = result[0]
            # Update existing record
            conn.execute(
                """
                UPDATE model_implementations
                SET file_path = ?, generation_date = ?, model_category = ?,
                    hardware_support = ?, primary_task = ?, cross_platform = ?
                WHERE implementation_id = ?
                """,
                [file_path, generation_date_str, model_category,
                 hardware_support_json, primary_task, cross_platform,
                 implementation_id]
            )
            logger.debug(f"Updated implementation metadata for {model_type} (ID: {implementation_id})")
            return implementation_id
        
        # Create new record
        conn.execute(
            """
            INSERT INTO model_implementations (
                model_type, file_path, generation_date, model_category,
                hardware_support, primary_task, cross_platform
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [model_type, file_path, generation_date_str, model_category,
             hardware_support_json, primary_task, cross_platform]
        )
        
        implementation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug(f"Stored implementation metadata for {model_type} (ID: {implementation_id})")
        return implementation_id
    except Exception as e:
        logger.error(f"Error storing implementation metadata for {model_type}: {e}")
        return None

def execute_query(query: str, params: List = None, db_path: str = None) -> List[Tuple]:
    """
    Execute a custom SQL query on the database.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query
        db_path: Path to the database file
        
    Returns:
        Query results as a list of tuples
    """
    if not DUCKDB_AVAILABLE:
        return []
    
    conn = get_db_connection(db_path)
    if not conn:
        return []
    
    try:
        result = conn.execute(query, params or []).fetchall()
        return result
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return []

def query_to_dataframe(query: str, params: List = None, db_path: str = None) -> Optional['pd.DataFrame']:
    """
    Execute a custom SQL query on the database and return results as a pandas DataFrame.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query
        db_path: Path to the database file
        
    Returns:
        Query results as a pandas DataFrame
    """
    if not DUCKDB_AVAILABLE:
        return None
    
    conn = get_db_connection(db_path)
    if not conn:
        return None
    
    try:
        # Use DuckDB's fetch_df method for efficient conversion to DataFrame
        result = conn.execute(query, params or []).fetch_df()
        return result
    except Exception as e:
        logger.error(f"Error executing query to dataframe: {e}")
        return None

def convert_json_to_db(json_file: str, category: str = None) -> bool:
    """
    Convert a JSON file to database records.
    
    Args:
        json_file: Path to the JSON file
        category: Category of the data (performance, hardware_compatibility, etc.)
        
    Returns:
        True if conversion was successful, False otherwise
    """
    if not DUCKDB_AVAILABLE:
        return False
    
    if not os.path.exists(json_file):
        logger.error(f"JSON file does not exist: {json_file}")
        return False
    
    # Determine category from filename if not provided
    if not category:
        filename = os.path.basename(json_file).lower()
        if "performance" in filename or "benchmark" in filename:
            category = "performance"
        elif "hardware" in filename and "compatibility" in filename:
            category = "hardware_compatibility"
        elif "web" in filename and "platform" in filename:
            category = "web_platform"
        elif "test" in filename and "result" in filename:
            category = "test_results"
        elif "integration" in filename:
            category = "integration_test"
        else:
            category = "unknown"
    
    try:
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert based on category
        if category == "performance":
            return _convert_performance_json(data)
        elif category == "hardware_compatibility":
            return _convert_hardware_compatibility_json(data)
        elif category == "web_platform":
            return _convert_web_platform_json(data)
        elif category == "test_results":
            return _convert_test_results_json(data)
        elif category == "integration_test":
            return _convert_integration_test_json(data)
        else:
            logger.warning(f"Unknown category for JSON file: {json_file}")
            return False
    except Exception as e:
        logger.error(f"Error converting JSON file {json_file}: {e}")
        return False

def _convert_performance_json(data: Dict) -> bool:
    """Convert performance benchmark data to database records."""
    # Implementation would parse the JSON data and insert into the database
    # This would vary based on the structure of the JSON file
    # For now, this is a placeholder
    return True

def _convert_hardware_compatibility_json(data: Dict) -> bool:
    """Convert hardware compatibility data to database records."""
    # Implementation would parse the JSON data and insert into the database
    # This would vary based on the structure of the JSON file
    # For now, this is a placeholder
    return True

def _convert_web_platform_json(data: Dict) -> bool:
    """Convert web platform benchmark data to database records."""
    # Implementation would parse the JSON data and insert into the database
    # This would vary based on the structure of the JSON file
    # For now, this is a placeholder
    return True

def _convert_test_results_json(data: Dict) -> bool:
    """Convert test results data to database records."""
    # Implementation would parse the JSON data and insert into the database
    # This would vary based on the structure of the JSON file
    # For now, this is a placeholder
    return True

def _convert_integration_test_json(data: Dict) -> bool:
    """Convert integration test data to database records."""
    # Implementation would parse the JSON data and insert into the database
    # This would vary based on the structure of the JSON file
    # For now, this is a placeholder
    return True

# Export public functions and constants
__all__ = [
    'DUCKDB_AVAILABLE',
    'DEPRECATE_JSON_OUTPUT',
    'BENCHMARK_DB_PATH',
    'get_db_connection',
    'close_all_connections',
    'create_test_run',
    'get_or_create_test_run',
    'complete_test_run',
    'get_or_create_model',
    'get_or_create_hardware',
    'store_performance_result',
    'store_hardware_compatibility',
    'store_web_platform_result',
    'store_test_result',
    'store_integration_test_result',
    'store_implementation_metadata',
    'execute_query',
    'query_to_dataframe',
    'convert_json_to_db'
]