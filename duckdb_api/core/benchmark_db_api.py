#!/usr/bin/env python
"""
Benchmark Database API for the IPFS Accelerate Python Framework.

This module provides a programmatic and REST API interface to the benchmark database,
allowing test runners to store results directly and providing query capabilities for
analysis and visualization.

Usage:
    # Start API server
    python benchmark_db_api.py --serve

    # Programmatic usage
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    api = BenchmarkDBAPI()
    api.store_performance_result(model_name="bert-base-uncased", hardware_type="cuda", ...)

    # WebNN/WebGPU benchmarking - store results directly
    from duckdb_api.core.benchmark_db_api import BenchmarkDatabase, store_benchmark_result
    db = BenchmarkDatabase("./benchmark_db.duckdb")
    store_benchmark_result(db, model_name="bert-base-uncased", hardware_type="webgpu_chrome", ...)
"""

import os
import sys
import json
import logging
import argparse
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


try:
    import duckdb
    import pandas as pd
    import fastapi
    from fastapi import FastAPI, HTTPException, Query, Body
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas fastapi uvicorn pydantic")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# BenchmarkDatabase class for direct access to the database
class BenchmarkDatabase:
    """
    Simple database connection class for WebNN/WebGPU benchmarking.
    This provides a lighter-weight alternative to the full BenchmarkDBAPI.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the benchmark database connection.
        
        Args:
            db_path: Path to the DuckDB database. If None, uses the BENCHMARK_DB_PATH 
                     environment variable or falls back to "./benchmark_db.duckdb"
        """
        # Get database path from environment variable if not provided
        if db_path is None:
            db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        self.db_path = db_path
        self.conn = None
        
        # Ensure database directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        logger.info(f"Initialized BenchmarkDatabase with path: {db_path}")
    
    def connect(self):
        """Create and return a connection to the database."""
        try:
            # Try newer DuckDB versions first
            try:
                return duckdb.connect(self.db_path, read_only=False, access_mode='automatic')
            except TypeError:
                # Fall back to older DuckDB versions
                return duckdb.connect(self.db_path, read_only=False)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def get_connection(self):
        """Get a database connection, creating one if needed."""
        if self.conn is None:
            self.conn = self.connect()
        return self.conn
    
    def close(self):
        """Close the database connection if open."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

# Function to store benchmark results from WebNN/WebGPU tests
def store_benchmark_result(
    db: BenchmarkDatabase,
    model_name: str,
    hardware_type: str,
    batch_size: int = 1,
    precision: Optional[str] = None,
    average_latency_ms: float = 0.0,
    throughput_items_per_second: float = 0.0,
    memory_mb: float = 0.0,
    is_simulation: bool = False,
    test_case: str = "default",
    browser: Optional[str] = None,
    device_name: Optional[str] = None,
    run_id: Optional[int] = None
) -> int:
    """
    Store a WebNN/WebGPU benchmark result in the database.
    
    Args:
        db: BenchmarkDatabase instance
        model_name: Name of the model
        hardware_type: Type of hardware (webnn_chrome, webgpu_firefox, etc.)
        batch_size: Batch size used for the benchmark
        precision: Precision used (fp32, int8, etc.)
        average_latency_ms: Average latency in milliseconds
        throughput_items_per_second: Throughput in items per second
        memory_mb: Memory usage in MB
        is_simulation: Whether this was a simulation or real hardware
        test_case: Name of the test case
        browser: Browser used (chrome, firefox, edge)
        device_name: Name of the device
        run_id: Optional test run ID
        
    Returns:
        ID of the stored result
    """
    # Connect to database
    conn = None
    try:
        conn = db.get_connection()
        
        # Create transaction for atomicity
        conn.execute("BEGIN TRANSACTION")
        
        # Append browser to device_name if provided but not already in device_name
        if browser and (not device_name or browser.lower() not in device_name.lower()):
            device_name = f"{browser} {device_name or ''}"
        
        # Ensure model exists
        model_id = _ensure_model_exists(conn, model_name)
        
        # Ensure hardware exists
        hardware_id = _ensure_hardware_exists(conn, hardware_type, device_name)
        
        # Create test run if run_id not provided
        if run_id is None:
            run_id = _create_test_run(
                conn,
                f"webnn_webgpu_benchmark_{model_name.replace('/', '_')}",
                "web_platform_performance",
                {"browser": browser, "is_simulation": is_simulation}
            )
        
        # Get next result_id
        max_id = conn.execute("SELECT COALESCE(MAX(result_id), 0) + 1 FROM performance_results").fetchone()[0]
        result_id = 1 if max_id is None else max_id
        
        # Prepare metrics JSON
        metrics = {
            "is_simulation": is_simulation,
            "browser": browser,
        }
        
        # Store performance result
        conn.execute("""
        INSERT INTO performance_results (
            result_id, run_id, model_id, hardware_id, test_case, batch_size, precision,
            average_latency_ms, throughput_items_per_second,
            memory_peak_mb, metrics
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            result_id, run_id, model_id, hardware_id, test_case, batch_size,
            precision, average_latency_ms, throughput_items_per_second,
            memory_mb, json.dumps(metrics)
        ])
        
        # Also store hardware compatibility result
        _store_compatibility_result(
            conn,
            run_id=run_id,
            model_id=model_id,
            hardware_id=hardware_id,
            is_compatible=True,  # We assume it's compatible if we got a result
            is_simulation=is_simulation
        )
        
        # Commit the transaction
        conn.execute("COMMIT")
        
        logger.info(f"Stored benchmark result for {model_name} on {hardware_type} (result_id: {result_id})")
        return result_id
    
    except Exception as e:
        # Rollback transaction on error
        if conn:
            try:
                conn.execute("ROLLBACK")
            except:
                pass
        logger.error(f"Error storing benchmark result: {e}")
        raise
    
    # Note: We don't close the connection here as it's managed by the BenchmarkDatabase instance

# Helper functions for store_benchmark_result
def _ensure_model_exists(conn, model_name: str) -> int:
    """Ensure a model exists in the database, adding it if not."""
    # Check if model exists
    result = conn.execute(
        "SELECT model_id FROM models WHERE model_name = ?", 
        [model_name]
    ).fetchone()
    
    if result:
        return result[0]
    
    # Model doesn't exist, try to infer family and modality
    model_family = None
    modality = None
    
    # Simple inference from model name
    lower_name = model_name.lower()
    if 'bert' in lower_name:
        model_family = 'embedding'
        modality = 'text'
    elif 't5' in lower_name:
        model_family = 'text_generation'
        modality = 'text'
    elif 'gpt' in lower_name or 'llama' in lower_name:
        model_family = 'text_generation'
        modality = 'text'
    elif 'clip' in lower_name:
        model_family = 'multimodal'
        modality = 'multimodal'
    elif 'vit' in lower_name:
        model_family = 'vision'
        modality = 'vision'
    elif 'whisper' in lower_name or 'wav2vec' in lower_name:
        model_family = 'audio'
        modality = 'audio'
    elif 'llava' in lower_name:
        model_family = 'multimodal'
        modality = 'multimodal'
    
    # Get next model_id
    max_id = conn.execute("SELECT COALESCE(MAX(model_id), 0) + 1 FROM models").fetchone()[0]
    model_id = 1 if max_id is None else max_id
    
    # Add model to database
    conn.execute(
        """
        INSERT INTO models (model_id, model_name, model_family, modality, source, version)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [model_id, model_name, model_family, modality, 'huggingface', '1.0']
    )
    
    logger.info(f"Added new model to database: {model_name} (ID: {model_id})")
    return model_id

def _ensure_hardware_exists(conn, hardware_type: str, device_name: str = None) -> int:
    """Ensure a hardware platform exists in the database, adding it if not."""
    # Use default device name if not provided
    if device_name is None:
        if 'webnn' in hardware_type.lower():
            device_name = 'WebNN Device'
        elif 'webgpu' in hardware_type.lower():
            device_name = 'WebGPU Device'
        elif hardware_type == 'cpu':
            device_name = 'CPU'
        elif hardware_type == 'cuda':
            device_name = 'NVIDIA GPU'
        elif hardware_type == 'rocm':
            device_name = 'AMD GPU'
        elif hardware_type == 'mps':
            device_name = 'Apple Silicon'
        elif hardware_type == 'openvino':
            device_name = 'OpenVINO'
        else:
            device_name = hardware_type.upper()
    
    # Extract browser if included in hardware_type
    browser = None
    if 'chrome' in hardware_type.lower():
        browser = 'Chrome'
    elif 'firefox' in hardware_type.lower():
        browser = 'Firefox'
    elif 'edge' in hardware_type.lower():
        browser = 'Edge'
    elif 'safari' in hardware_type.lower():
        browser = 'Safari'
    
    # Add browser to device name if not already included
    if browser and browser.lower() not in device_name.lower():
        device_name = f"{browser} {device_name}"
    
    # Check if hardware exists
    result = conn.execute(
        "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?",
        [hardware_type, device_name]
    ).fetchone()
    
    if result:
        return result[0]
    
    # Hardware doesn't exist, add it
    # Get next hardware_id
    max_id = conn.execute("SELECT COALESCE(MAX(hardware_id), 0) + 1 FROM hardware_platforms").fetchone()[0]
    hardware_id = 1 if max_id is None else max_id
    
    # Determine platform based on hardware_type
    platform = None
    if 'webnn' in hardware_type.lower():
        platform = 'WebNN'
    elif 'webgpu' in hardware_type.lower():
        platform = 'WebGPU'
    else:
        platform = hardware_type.upper()
    
    conn.execute(
        """
        INSERT INTO hardware_platforms (
            hardware_id, hardware_type, device_name, platform, platform_version, 
            driver_version, memory_gb, compute_units
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [hardware_id, hardware_type, device_name, platform, 'unknown', 'unknown', 0, 0]
    )
    
    logger.info(f"Added new hardware to database: {hardware_type} - {device_name} (ID: {hardware_id})")
    return hardware_id

def _create_test_run(conn, test_name: str, test_type: str, metadata: Dict = None) -> int:
    """Create a new test run entry in the database."""
    # Get next run_id
    max_id = conn.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM test_runs").fetchone()[0]
    run_id = 1 if max_id is None else max_id
    
    # Current timestamp
    now = datetime.datetime.now()
    
    # Insert test run
    conn.execute(
        """
        INSERT INTO test_runs (
            run_id, test_name, test_type, started_at, completed_at, 
            execution_time_seconds, success, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [run_id, test_name, test_type, now, now, 0, True, json.dumps(metadata or {})]
    )
    
    logger.debug(f"Created new test run: {test_name} (ID: {run_id})")
    return run_id

def _store_compatibility_result(conn, run_id: int, model_id: int, hardware_id: int, 
                               is_compatible: bool, is_simulation: bool = False) -> int:
    """Store hardware compatibility result in the database."""
    # Get next compatibility_id
    max_id = conn.execute("SELECT COALESCE(MAX(compatibility_id), 0) + 1 FROM hardware_compatibility").fetchone()[0]
    compatibility_id = 1 if max_id is None else max_id
    
    # Prepare metadata
    metadata = {"is_simulation": is_simulation}
    
    # Insert compatibility result
    conn.execute(
        """
        INSERT INTO hardware_compatibility (
            compatibility_id, run_id, model_id, hardware_id, is_compatible,
            detection_success, initialization_success, compatibility_score, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            compatibility_id, run_id, model_id, hardware_id, is_compatible,
            True, is_compatible, 1.0 if is_compatible else 0.0, json.dumps(metadata)
        ]
    )
    
    return compatibility_id

# Models for API requests and responses
class PerformanceResult(BaseModel):
    model_name: str
    hardware_type: str
    device_name: Optional[str] = None
    batch_size: int = 1
    precision: str = "fp32"
    test_case: str = "default"
    throughput: float
    latency_avg: float
    latency_p90: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    memory_peak: Optional[float] = None
    total_time_seconds: Optional[float] = None
    iterations: Optional[int] = None
    warmup_iterations: Optional[int] = None
    run_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
class HardwareCompatibility(BaseModel):
    model_name: str
    hardware_type: str
    device_name: Optional[str] = None
    is_compatible: bool
    detection_success: bool = True
    initialization_success: bool = True
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    suggested_fix: Optional[str] = None
    workaround_available: bool = False
    compatibility_score: Optional[float] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class IntegrationTestResult(BaseModel):
    test_module: str
    test_class: Optional[str] = None
    test_name: str
    status: str  # 'pass', 'fail', 'error', 'skip'
    execution_time_seconds: Optional[float] = None
    hardware_type: Optional[str] = None
    device_name: Optional[str] = None
    model_name: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    assertions: Optional[List[Dict[str, Any]]] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    sql: str
    parameters: Optional[Dict[str, Any]] = None

class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    result_id: Optional[str] = None

class BenchmarkDBAPI:
    """
    API interface to the benchmark database for storing and querying results.
    """
    
    def __init__(self, db_path: str = None, debug: bool = False):
        """
        Initialize the benchmark database API.
        
        Args:
            db_path: Path to the DuckDB database. If None, uses the BENCHMARK_DB_PATH 
                     environment variable or falls back to "./benchmark_db.duckdb"
            debug: Enable debug logging
        """
        # Get database path from environment variable if not provided
        if db_path is None:
            db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Ensure database exists
        self._ensure_db_exists()
        
        logger.info(f"Initialized BenchmarkDBAPI with DB: {db_path}")
    
    def _ensure_db_exists(self):
        """
        Ensure that the database exists and has the expected schema.
        If not, initialize it with the schema creation script.
        """
        db_file = Path(self.db_path)
        
        # Check if database file exists
        if not db_file.exists():
            logger.info(f"Database file does not exist. Creating new database at {self.db_path}")
            
            # Create parent directories if they don't exist
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create in a safe way (directly with connection)
            conn = None
            try:
                # Try to connect directly and create schema
                conn = duckdb.connect(self.db_path)
                
                # Import and run the create_benchmark_schema script
                # Check multiple possible locations for the schema script
                schema_paths = [
                    str(Path(__file__).parent / "scripts" / "create_benchmark_schema.py"),
                    str(Path(__file__).parent / "scripts" / "benchmark_db" / "create_benchmark_schema.py"),
                    "scripts/create_benchmark_schema.py",
                    "test/scripts/create_benchmark_schema.py"
                ]
                
                schema_script = None
                for path in schema_paths:
                    if Path(path).exists():
                        schema_script = path
                        break
                
                if schema_script:
                    logger.info(f"Creating schema using script: {schema_script}")
                    # Close current connection before running script
                    if conn:
                        conn.close()
                        conn = None
                    
                    import subprocess
                    result = subprocess.run([sys.executable, schema_script, "--output", self.db_path, "--sample-data"], 
                                        capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.error(f"Error running schema script: {result.stderr}")
                        # Re-open connection for fallback
                        conn = duckdb.connect(self.db_path)
                        self._create_minimal_schema(conn)
                else:
                    logger.warning(f"Schema script not found. Checked paths: {schema_paths}. Creating minimal schema.")
                    self._create_minimal_schema(conn)
            
            except Exception as e:
                logger.error(f"Error creating database schema: {e}")
                if conn:
                    self._create_minimal_schema(conn)
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception as close_error:
                        logger.error(f"Error closing database connection: {close_error}")
    
    def _create_minimal_schema(self, conn=None):
        """
        Create a minimal schema if the full schema creation script is not available.
        
        Args:
            conn: Optional database connection to use. If None, a new connection will be created.
        """
        logger.info("Creating minimal schema")
        
        # Connect to database if not provided
        close_conn = False
        if conn is None:
            conn = duckdb.connect(self.db_path)
            close_conn = True
        
        try:
            # Start a transaction for consistency
            conn.execute("BEGIN TRANSACTION")
            
            # Create basic tables
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_platforms (
                hardware_id INTEGER PRIMARY KEY,
                hardware_type VARCHAR NOT NULL,
                device_name VARCHAR,
                platform VARCHAR,
                platform_version VARCHAR,
                driver_version VARCHAR,
                memory_gb FLOAT,
                compute_units INTEGER,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_family VARCHAR,
                modality VARCHAR,
                source VARCHAR,
                version VARCHAR,
                parameters_million FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id INTEGER PRIMARY KEY,
                test_name VARCHAR NOT NULL,
                test_type VARCHAR NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                execution_time_seconds FLOAT,
                success BOOLEAN,
                git_commit VARCHAR,
                git_branch VARCHAR,
                command_line VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create tables without foreign keys to avoid errors
            conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_results (
                result_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_id INTEGER NOT NULL,
                hardware_id INTEGER NOT NULL,
                test_case VARCHAR NOT NULL,
                batch_size INTEGER DEFAULT 1,
                precision VARCHAR,
                total_time_seconds FLOAT,
                average_latency_ms FLOAT,
                throughput_items_per_second FLOAT,
                memory_peak_mb FLOAT,
                iterations INTEGER,
                warmup_iterations INTEGER,
                metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_compatibility (
                compatibility_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_id INTEGER NOT NULL,
                hardware_id INTEGER NOT NULL,
                is_compatible BOOLEAN NOT NULL,
                detection_success BOOLEAN NOT NULL,
                initialization_success BOOLEAN NOT NULL,
                error_message VARCHAR,
                error_type VARCHAR,
                suggested_fix VARCHAR,
                workaround_available BOOLEAN,
                compatibility_score FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Commit the transaction
            conn.execute("COMMIT")
            
            logger.info("Minimal schema created successfully")
        except Exception as e:
            # Rollback in case of error
            try:
                conn.execute("ROLLBACK")
                logger.warning("Schema creation failed, rolled back transaction")
            except Exception as rollback_e:
                logger.error(f"Error rolling back transaction: {rollback_e}")
            
            logger.error(f"Error creating minimal schema: {e}")
            
            # Try minimal fallback schema as a last resort
            try:
                self._create_fallback_schema(conn)
            except Exception as fallback_e:
                logger.error(f"Error creating fallback schema: {fallback_e}")
        finally:
            # Only close the connection if we created it
            if close_conn and conn:
                try:
                    conn.close()
                except Exception as close_e:
                    logger.error(f"Error closing connection: {close_e}")
    
    def _create_fallback_schema(self, conn):
        """
        Create a very minimal schema as a last resort.
        
        Args:
            conn: Database connection
        """
        logger.warning("Creating fallback minimal schema without transaction")
        
        try:
            # Create tables individually with minimal fields
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_platforms (
                hardware_id INTEGER PRIMARY KEY,
                hardware_type VARCHAR NOT NULL
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                model_name VARCHAR NOT NULL
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id INTEGER PRIMARY KEY,
                test_name VARCHAR NOT NULL
            )
            """)
            
            logger.info("Fallback schema created successfully")
        except Exception as e:
            logger.error(f"Error creating fallback schema: {e}")
    
    def _get_connection(self):
        """Get a connection to the database with appropriate settings."""
        try:
            # Use parameters compatible with the installed DuckDB version
            try:
                # Try newer DuckDB versions
                return duckdb.connect(self.db_path, read_only=False, access_mode='automatic')
            except TypeError:
                # Fall back to older DuckDB versions
                return duckdb.connect(self.db_path, read_only=False)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _ensure_model_exists(self, conn, model_name: str) -> int:
        """
        Ensure that a model exists in the database, adding it if not.
        
        Args:
            conn: Database connection
            model_name: Name of the model
            
        Returns:
            model_id: ID of the model in the database
        """
        # Check if model exists
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            [model_name]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Model doesn't exist, try to infer family and modality
        model_family = None
        modality = None
        
        # Simple inference from model name
        lower_name = model_name.lower()
        if 'bert' in lower_name:
            model_family = 'bert'
            modality = 'text'
        elif 't5' in lower_name:
            model_family = 't5'
            modality = 'text'
        elif 'gpt' in lower_name or 'llama' in lower_name:
            model_family = 'llm'
            modality = 'text'
        elif 'clip' in lower_name or 'vit' in lower_name:
            model_family = 'vision'
            modality = 'image'
        elif 'whisper' in lower_name or 'wav2vec' in lower_name:
            model_family = 'audio'
            modality = 'audio'
        elif 'llava' in lower_name:
            model_family = 'multimodal'
            modality = 'multimodal'
        
        # Add model to database
        # Get next model_id
        max_id = conn.execute("SELECT MAX(model_id) FROM models").fetchone()[0]
        model_id = 1 if max_id is None else max_id + 1
        
        conn.execute(
            """
            INSERT INTO models (model_id, model_name, model_family, modality, source, version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [model_id, model_name, model_family, modality, 'unknown', '1.0']
        )
        
        logger.info(f"Added new model to database: {model_name} (ID: {model_id})")
        return model_id
    
    def _ensure_hardware_exists(self, conn, hardware_type: str, device_name: str = None) -> int:
        """
        Ensure that a hardware platform exists in the database, adding it if not.
        
        Args:
            conn: Database connection
            hardware_type: Type of hardware (cpu, cuda, rocm, etc.)
            device_name: Name of the device
            
        Returns:
            hardware_id: ID of the hardware in the database
        """
        # Use default device name if not provided
        if device_name is None:
            if hardware_type == 'cpu':
                device_name = 'CPU'
            elif hardware_type == 'cuda':
                device_name = 'NVIDIA GPU'
            elif hardware_type == 'rocm':
                device_name = 'AMD GPU'
            elif hardware_type == 'mps':
                device_name = 'Apple Silicon'
            elif hardware_type == 'openvino':
                device_name = 'OpenVINO'
            elif hardware_type == 'webnn':
                device_name = 'WebNN'
            elif hardware_type == 'webgpu':
                device_name = 'WebGPU'
            else:
                device_name = hardware_type.upper()
        
        # Check if hardware exists
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?",
            [hardware_type, device_name]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Hardware doesn't exist, add it
        # Get next hardware_id
        max_id = conn.execute("SELECT MAX(hardware_id) FROM hardware_platforms").fetchone()[0]
        hardware_id = 1 if max_id is None else max_id + 1
        
        conn.execute(
            """
            INSERT INTO hardware_platforms (
                hardware_id, hardware_type, device_name, platform, platform_version, 
                driver_version, memory_gb, compute_units
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [hardware_id, hardware_type, device_name, 'unknown', 'unknown', 'unknown', 0, 0]
        )
        
        logger.info(f"Added new hardware to database: {hardware_type} - {device_name} (ID: {hardware_id})")
        return hardware_id
    
    def _create_test_run(self, conn, test_name: str, test_type: str, metadata: Dict = None) -> int:
        """
        Create a new test run entry in the database.
        
        Args:
            conn: Database connection
            test_name: Name of the test
            test_type: Type of test (performance, hardware, compatibility, integration)
            metadata: Additional metadata for the test run
            
        Returns:
            run_id: ID of the test run in the database
        """
        # Get next run_id
        max_id = conn.execute("SELECT MAX(run_id) FROM test_runs").fetchone()[0]
        run_id = 1 if max_id is None else max_id + 1
        
        # Current timestamp
        now = datetime.datetime.now()
        
        # Insert test run
        conn.execute(
            """
            INSERT INTO test_runs (
                run_id, test_name, test_type, started_at, completed_at, 
                execution_time_seconds, success, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, test_name, test_type, now, now, 0, True, json.dumps(metadata or {})]
        )
        
        logger.debug(f"Created new test run: {test_name} (ID: {run_id})")
        return run_id
    
    def store_performance_result(self, result: Union[PerformanceResult, Dict]) -> str:
        """
        Store a performance benchmark result in the database.
        
        Args:
            result: Performance benchmark result data
            
        Returns:
            result_id: ID of the stored result
        """
        # Convert dict to PerformanceResult if needed
        if isinstance(result, dict):
            result = PerformanceResult(**result)
        
        # Validate required fields
        if not result.model_name:
            raise ValueError("model_name is required")
        if not result.hardware_type:
            raise ValueError("hardware_type is required")
        if result.throughput is None:
            raise ValueError("throughput is required")
        if result.latency_avg is None:
            raise ValueError("latency_avg is required")
        
        conn = self._get_connection()
        try:
            # Start a transaction for data consistency
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Get or create model
                model_id = self._ensure_model_exists(conn, result.model_name)
                
                # Get or create hardware
                hardware_id = self._ensure_hardware_exists(conn, result.hardware_type, result.device_name)
                
                # Create test run if run_id not provided
                if result.run_id:
                    # Check if run exists
                    run_exists = conn.execute(
                        "SELECT COUNT(*) FROM test_runs WHERE run_id = ?",
                        [result.run_id]
                    ).fetchone()[0] > 0
                    
                    if not run_exists:
                        logger.warning(f"Test run with ID {result.run_id} not found, creating new run")
                        run_id = self._create_test_run(
                            conn,
                            f"performance_benchmark_{result.model_name}",
                            "performance",
                            {"source": "api", "model": result.model_name, "hardware": result.hardware_type}
                        )
                    else:
                        run_id = result.run_id
                else:
                    run_id = self._create_test_run(
                        conn,
                        f"performance_benchmark_{result.model_name}",
                        "performance",
                        {"source": "api", "model": result.model_name, "hardware": result.hardware_type}
                    )
                
                # Get next result_id
                max_id = conn.execute("SELECT MAX(result_id) FROM performance_results").fetchone()[0]
                result_id = 1 if max_id is None else max_id + 1
                
                # Store performance result
                conn.execute(
                    """
                    INSERT INTO performance_results (
                        result_id, run_id, model_id, hardware_id, test_case, batch_size, precision,
                        total_time_seconds, average_latency_ms, throughput_items_per_second,
                        memory_peak_mb, iterations, warmup_iterations, metrics
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        result_id, run_id, model_id, hardware_id, result.test_case, result.batch_size,
                        result.precision, result.total_time_seconds, result.latency_avg,
                        result.throughput, result.memory_peak, result.iterations,
                        result.warmup_iterations, json.dumps(result.metrics or {})
                    ]
                )
                
                # Commit the transaction
                conn.execute("COMMIT")
                logger.info(f"Stored performance result for {result.model_name} on {result.hardware_type} (ID: {result_id})")
                return str(result_id)
                
            except Exception as inner_ex:
                # Rollback on error
                try:
                    conn.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
                # Re-raise the original exception
                raise inner_ex
            
        except Exception as e:
            logger.error(f"Error storing performance result: {e}")
            raise
        finally:
            conn.close()
    
    def store_compatibility_result(self, result: Union[HardwareCompatibility, Dict]) -> str:
        """
        Store a hardware compatibility result in the database.
        
        Args:
            result: Hardware compatibility result data
            
        Returns:
            compatibility_id: ID of the stored result
        """
        # Convert dict to HardwareCompatibility if needed
        if isinstance(result, dict):
            result = HardwareCompatibility(**result)
        
        # Validate required fields
        if not result.model_name:
            raise ValueError("model_name is required")
        if not result.hardware_type:
            raise ValueError("hardware_type is required")
        if result.is_compatible is None:
            raise ValueError("is_compatible is required")
        
        conn = self._get_connection()
        try:
            # Start a transaction for data consistency
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Get or create model
                model_id = self._ensure_model_exists(conn, result.model_name)
                
                # Get or create hardware
                hardware_id = self._ensure_hardware_exists(conn, result.hardware_type, result.device_name)
                
                # Create test run if run_id not provided
                if result.run_id:
                    # Check if run exists
                    run_exists = conn.execute(
                        "SELECT COUNT(*) FROM test_runs WHERE run_id = ?",
                        [result.run_id]
                    ).fetchone()[0] > 0
                    
                    if not run_exists:
                        logger.warning(f"Test run with ID {result.run_id} not found, creating new run")
                        run_id = self._create_test_run(
                            conn,
                            f"hardware_compatibility_{result.model_name}",
                            "hardware",
                            {"source": "api", "model": result.model_name, "hardware": result.hardware_type}
                        )
                    else:
                        run_id = result.run_id
                else:
                    run_id = self._create_test_run(
                        conn,
                        f"hardware_compatibility_{result.model_name}",
                        "hardware",
                        {"source": "api", "model": result.model_name, "hardware": result.hardware_type}
                    )
                
                # Get next compatibility_id
                max_id = conn.execute("SELECT MAX(compatibility_id) FROM hardware_compatibility").fetchone()[0]
                compatibility_id = 1 if max_id is None else max_id + 1
                
                # Store compatibility result
                conn.execute(
                    """
                    INSERT INTO hardware_compatibility (
                        compatibility_id, run_id, model_id, hardware_id, is_compatible,
                        detection_success, initialization_success, error_message, error_type,
                        suggested_fix, workaround_available, compatibility_score, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        compatibility_id, run_id, model_id, hardware_id, result.is_compatible,
                        result.detection_success, result.initialization_success, result.error_message,
                        result.error_type, result.suggested_fix, result.workaround_available,
                        result.compatibility_score, json.dumps(result.metadata or {})
                    ]
                )
                
                # Commit the transaction
                conn.execute("COMMIT")
                logger.info(f"Stored compatibility result for {result.model_name} on {result.hardware_type} (ID: {compatibility_id})")
                return str(compatibility_id)
                
            except Exception as inner_ex:
                # Rollback on error
                try:
                    conn.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
                # Re-raise the original exception
                raise inner_ex
            
        except Exception as e:
            logger.error(f"Error storing compatibility result: {e}")
            raise
        finally:
            conn.close()
    
    def store_integration_test_result(self, result: Union[IntegrationTestResult, Dict]) -> str:
        """
        Store an integration test result in the database.
        
        Args:
            result: Integration test result data
            
        Returns:
            test_result_id: ID of the stored result
        """
        # Convert dict to IntegrationTestResult if needed
        if isinstance(result, dict):
            result = IntegrationTestResult(**result)
        
        # Validate required fields
        if not result.test_module:
            raise ValueError("test_module is required")
        if not result.test_name:
            raise ValueError("test_name is required")
        if not result.status:
            raise ValueError("status is required")
        
        conn = self._get_connection()
        try:
            # Start a transaction for data consistency
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Get model_id if model_name provided
                model_id = None
                if result.model_name:
                    model_id = self._ensure_model_exists(conn, result.model_name)
                
                # Get hardware_id if hardware_type provided
                hardware_id = None
                if result.hardware_type:
                    hardware_id = self._ensure_hardware_exists(conn, result.hardware_type, result.device_name)
                
                # Create test run if run_id not provided
                if result.run_id:
                    # Check if run exists
                    run_exists = conn.execute(
                        "SELECT COUNT(*) FROM test_runs WHERE run_id = ?",
                        [result.run_id]
                    ).fetchone()[0] > 0
                    
                    if not run_exists:
                        logger.warning(f"Test run with ID {result.run_id} not found, creating new run")
                        run_id = self._create_test_run(
                            conn,
                            f"integration_test_{result.test_module}",
                            "integration",
                            {"source": "api", "test_module": result.test_module}
                        )
                    else:
                        run_id = result.run_id
                else:
                    run_id = self._create_test_run(
                        conn,
                        f"integration_test_{result.test_module}",
                        "integration",
                        {"source": "api", "test_module": result.test_module}
                    )
                
                # Get next test_result_id
                max_id = conn.execute("SELECT MAX(test_result_id) FROM integration_test_results").fetchone()[0]
                test_result_id = 1 if max_id is None else max_id + 1
                
                # Store integration test result
                conn.execute(
                    """
                    INSERT INTO integration_test_results (
                        test_result_id, run_id, test_module, test_class, test_name, status,
                        execution_time_seconds, hardware_id, model_id, error_message, error_traceback, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        test_result_id, run_id, result.test_module, result.test_class, result.test_name,
                        result.status, result.execution_time_seconds, hardware_id, model_id,
                        result.error_message, result.error_traceback, json.dumps(result.metadata or {})
                    ]
                )
                
                # Store assertions if provided
                if result.assertions:
                    for i, assertion in enumerate(result.assertions):
                        assertion_id = i + 1
                        conn.execute(
                            """
                            INSERT INTO integration_test_assertions (
                                assertion_id, test_result_id, assertion_name, passed,
                                expected_value, actual_value, message
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                assertion_id, test_result_id, assertion.get('name', f'assertion_{i}'),
                                assertion.get('passed', False), assertion.get('expected', ''),
                                assertion.get('actual', ''), assertion.get('message', '')
                            ]
                        )
                
                # Commit the transaction
                conn.execute("COMMIT")
                logger.info(f"Stored integration test result for {result.test_module}.{result.test_name} (ID: {test_result_id})")
                return str(test_result_id)
                
            except Exception as inner_ex:
                # Rollback on error
                try:
                    conn.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
                # Re-raise the original exception
                raise inner_ex
                
        except Exception as e:
            logger.error(f"Error storing integration test result: {e}")
            raise
        finally:
            conn.close()
    
    def query(self, sql: str, parameters: Dict = None) -> pd.DataFrame:
        """
        Execute a SQL query against the database.
        
        Args:
            sql: SQL query string
            parameters: Parameters for the query
            
        Returns:
            DataFrame with the query results
        """
        conn = self._get_connection()
        try:
            # Execute query
            if parameters:
                result = conn.execute(sql, parameters)
            else:
                result = conn.execute(sql)
            
            # Convert to DataFrame
            df = result.fetch_df()
            
            logger.debug(f"Query executed: {sql}")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            conn.close()
    
    def get_model_hardware_compatibility(self, model_name: Optional[str] = None,
                                        hardware_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get model-hardware compatibility data.
        
        Args:
            model_name: Filter by model name (optional)
            hardware_type: Filter by hardware type (optional)
            
        Returns:
            DataFrame with compatibility data
        """
        sql = """
        SELECT 
            m.model_name,
            m.model_family,
            hp.hardware_type,
            hp.device_name,
            COUNT(CASE WHEN hc.is_compatible THEN 1 END) AS compatible_count,
            COUNT(CASE WHEN NOT hc.is_compatible THEN 1 END) AS incompatible_count,
            AVG(CASE WHEN hc.compatibility_score IS NOT NULL THEN hc.compatibility_score ELSE 
                CASE WHEN hc.is_compatible THEN 1.0 ELSE 0.0 END END) AS avg_compatibility_score,
            MAX(hc.created_at) AS last_tested
        FROM 
            hardware_compatibility hc
        JOIN 
            models m ON hc.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON hc.hardware_id = hp.hardware_id
        """
        
        conditions = []
        parameters = {}
        
        if model_name:
            conditions.append("m.model_name = :model_name")
            parameters['model_name'] = model_name
        
        if hardware_type:
            conditions.append("hp.hardware_type = :hardware_type")
            parameters['hardware_type'] = hardware_type
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += """
        GROUP BY 
            m.model_name, m.model_family, hp.hardware_type, hp.device_name
        """
        
        return self.query(sql, parameters)
    
    def get_performance_metrics(self, model_name: Optional[str] = None,
                               hardware_type: Optional[str] = None,
                               batch_size: Optional[int] = None,
                               precision: Optional[str] = None,
                               latest_only: bool = True) -> pd.DataFrame:
        """
        Get performance metrics data.
        
        Args:
            model_name: Filter by model name (optional)
            hardware_type: Filter by hardware type (optional)
            batch_size: Filter by batch size (optional)
            precision: Filter by precision (optional)
            latest_only: Return only the latest results for each model-hardware combination
            
        Returns:
            DataFrame with performance metrics
        """
        sql = """
        SELECT 
            m.model_name,
            m.model_family,
            hp.hardware_type,
            hp.device_name,
            pr.batch_size,
            pr.precision,
            pr.test_case,
            pr.average_latency_ms,
            pr.throughput_items_per_second,
            pr.memory_peak_mb,
            pr.created_at
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        """
        
        conditions = []
        parameters = {}
        
        if model_name:
            conditions.append("m.model_name = :model_name")
            parameters['model_name'] = model_name
        
        if hardware_type:
            conditions.append("hp.hardware_type = :hardware_type")
            parameters['hardware_type'] = hardware_type
        
        if batch_size:
            conditions.append("pr.batch_size = :batch_size")
            parameters['batch_size'] = batch_size
            
        if precision:
            conditions.append("pr.precision = :precision")
            parameters['precision'] = precision
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        if latest_only:
            sql = f"""
            WITH ranked_results AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size, pr.precision 
                                    ORDER BY pr.created_at DESC) as rn
                FROM ({sql}) as base
            )
            SELECT * FROM ranked_results WHERE rn = 1
            """
        
        return self.query(sql, parameters)
    
    def get_integration_test_summary(self, test_module: Optional[str] = None) -> pd.DataFrame:
        """
        Get integration test summary.
        
        Args:
            test_module: Filter by test module (optional)
            
        Returns:
            DataFrame with integration test summary
        """
        sql = """
        SELECT 
            test_module,
            COUNT(*) as total_tests,
            COUNT(CASE WHEN status = 'pass' THEN 1 END) as passed,
            COUNT(CASE WHEN status = 'fail' THEN 1 END) as failed,
            COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
            COUNT(CASE WHEN status = 'skip' THEN 1 END) as skipped,
            MAX(created_at) as last_run
        FROM 
            integration_test_results
        """
        
        parameters = {}
        if test_module:
            sql += " WHERE test_module = :test_module"
            parameters['test_module'] = test_module
        
        sql += " GROUP BY test_module"
        
        return self.query(sql, parameters)
    
    def get_hardware_list(self) -> pd.DataFrame:
        """Get a list of available hardware platforms."""
        sql = """
        SELECT 
            hardware_type,
            device_name,
            COUNT(*) as usage_count
        FROM 
            hardware_platforms
        GROUP BY 
            hardware_type, device_name
        ORDER BY 
            usage_count DESC
        """
        return self.query(sql)
    
    def get_model_list(self) -> pd.DataFrame:
        """Get a list of available models."""
        sql = """
        SELECT 
            model_name,
            model_family,
            modality,
            COUNT(*) as usage_count
        FROM 
            models
        GROUP BY 
            model_name, model_family, modality
        ORDER BY 
            usage_count DESC
        """
        return self.query(sql)
    
    def store_benchmark_results(self, results: Dict[str, Any]) -> str:
        """
        Store benchmark results in the database.
        
        This method is a wrapper around store_performance_result to maintain compatibility
        with benchmark runners that expect this method.
        
        Args:
            results: Dictionary with benchmark results
            
        Returns:
            result_id: ID of the stored result
        """
        logger.info("store_benchmark_results called - forwarding to store_performance_result")
        
        # Extract relevant fields from results
        try:
            # Basic fields
            model_name = results.get("model_name")
            if not model_name:
                model_name = results.get("model", "unknown_model")
                
            hardware_type = results.get("hardware_type")
            if not hardware_type:
                hardware_type = results.get("hardware", "cpu")
                
            # Extract performance metrics
            batch_size = results.get("batch_size", 1)
            
            # Look for metrics in different possible locations in the results dictionary
            throughput = None
            latency_avg = None
            memory_peak = None
            
            # Try direct access first
            throughput = results.get("throughput_items_per_second", results.get("throughput"))
            latency_avg = results.get("average_latency_ms", results.get("latency_avg", results.get("latency")))
            memory_peak = results.get("memory_peak_mb", results.get("memory_peak", results.get("memory")))
            
            # Try nested dictionaries if direct access fails
            if "metrics" in results:
                metrics = results["metrics"]
                if throughput is None:
                    throughput = metrics.get("throughput_items_per_second", metrics.get("throughput"))
                if latency_avg is None:
                    latency_avg = metrics.get("average_latency_ms", metrics.get("latency_avg", metrics.get("latency")))
                if memory_peak is None:
                    memory_peak = metrics.get("memory_peak_mb", metrics.get("memory_peak", metrics.get("memory")))
                    
            # If batch_sizes exists, try to extract metrics from there
            if "batch_sizes" in results and isinstance(results["batch_sizes"], dict):
                batch_data = results["batch_sizes"].get(str(batch_size))
                if batch_data:
                    if throughput is None:
                        throughput = batch_data.get("throughput_items_per_second", batch_data.get("throughput"))
                    if latency_avg is None:
                        latency_avg = batch_data.get("average_latency_ms", batch_data.get("latency_avg", batch_data.get("latency")))
                    if memory_peak is None:
                        memory_peak = batch_data.get("memory_peak_mb", batch_data.get("memory_peak", batch_data.get("memory")))
            
            # Use reasonable defaults if values are still None
            if throughput is None:
                throughput = 0.0
                logger.warning(f"No throughput found in results for {model_name} on {hardware_type}, using default 0.0")
            
            if latency_avg is None:
                latency_avg = 0.0
                logger.warning(f"No latency found in results for {model_name} on {hardware_type}, using default 0.0")
            
            if memory_peak is None:
                memory_peak = 0.0
                logger.warning(f"No memory usage found in results for {model_name} on {hardware_type}, using default 0.0")
            
            # Create performance result
            performance_result = {
                "model_name": model_name,
                "hardware_type": hardware_type,
                "device_name": results.get("device_name"),
                "batch_size": batch_size,
                "precision": results.get("precision", "fp32"),
                "test_case": results.get("test_case", "default"),
                "throughput": float(throughput),
                "latency_avg": float(latency_avg),
                "memory_peak": float(memory_peak),
                "total_time_seconds": results.get("total_time_seconds"),
                "iterations": results.get("iterations"),
                "warmup_iterations": results.get("warmup_iterations"),
                "metrics": results.get("metrics"),
            }
            
            # Store performance result
            return self.store_performance_result(performance_result)
            
        except Exception as e:
            logger.error(f"Error parsing benchmark results for store_benchmark_results: {e}")
            logger.error(f"Results keys: {list(results.keys())}")
            
            # Try to store with minimal data
            try:
                performance_result = {
                    "model_name": results.get("model_name", results.get("model", "unknown_model")),
                    "hardware_type": results.get("hardware_type", results.get("hardware", "cpu")),
                    "throughput": 0.0,
                    "latency_avg": 0.0,
                }
                return self.store_performance_result(performance_result)
            except Exception as inner_e:
                logger.error(f"Failed to store minimal benchmark results: {inner_e}")
                raise
    
    def get_performance_comparison(self, model_name: str, metric: str = "throughput") -> pd.DataFrame:
        """
        Get performance comparison across hardware platforms for a specific model.
        
        Args:
            model_name: Model name to compare
            metric: Metric to compare ("throughput", "latency", "memory")
            
        Returns:
            DataFrame with performance comparison
        """
        metric_column = "throughput_items_per_second"
        if metric.lower() == "latency":
            metric_column = "average_latency_ms"
        elif metric.lower() == "memory":
            metric_column = "memory_peak_mb"
        
        sql = f"""
        WITH latest_results AS (
            SELECT 
                m.model_name,
                hp.hardware_type,
                hp.device_name,
                pr.batch_size,
                pr.precision,
                pr.{metric_column} as metric_value,
                ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size, pr.precision 
                                ORDER BY pr.created_at DESC) as rn
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                m.model_name = :model_name
        )
        SELECT 
            model_name,
            hardware_type,
            device_name,
            batch_size,
            precision,
            metric_value
        FROM 
            latest_results
        WHERE 
            rn = 1
        ORDER BY 
            metric_value {"DESC" if metric.lower() == "throughput" else "ASC"}
        """
        
        return self.query(sql, {"model_name": model_name})

# Create FastAPI app if module is run directly
def create_app():
    """Create FastAPI app for the benchmark database API."""
    app = FastAPI(
        title="Benchmark Database API",
        description="API for storing and querying benchmark results",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create API instance
    api = BenchmarkDBAPI()
    
    # Root endpoint
    @app.get("/")
    def read_root():
        return {"message": "Benchmark Database API", "version": "0.1.0"}
    
    # Health check
    @app.get("/health")
    def health_check():
        return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}
    
    # Performance endpoints
    @app.post("/performance", response_model=SuccessResponse)
    def store_performance(result: PerformanceResult):
        result_id = api.store_performance_result(result)
        return SuccessResponse(success=True, message="Performance result stored successfully", result_id=result_id)
    
    @app.get("/performance")
    def get_performance(
        model_name: Optional[str] = None,
        hardware_type: Optional[str] = None,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        latest_only: bool = True
    ):
        df = api.get_performance_metrics(model_name, hardware_type, batch_size, precision, latest_only)
        return df.to_dict(orient="records")
    
    @app.get("/performance/comparison/{model_name}")
    def get_performance_comparison(
        model_name: str,
        metric: str = Query("throughput", enum=["throughput", "latency", "memory"])
    ):
        df = api.get_performance_comparison(model_name, metric)
        return df.to_dict(orient="records")
    
    # Compatibility endpoints
    @app.post("/compatibility", response_model=SuccessResponse)
    def store_compatibility(result: HardwareCompatibility):
        result_id = api.store_compatibility_result(result)
        return SuccessResponse(success=True, message="Compatibility result stored successfully", result_id=result_id)
    
    @app.get("/compatibility")
    def get_compatibility(
        model_name: Optional[str] = None,
        hardware_type: Optional[str] = None
    ):
        df = api.get_model_hardware_compatibility(model_name, hardware_type)
        return df.to_dict(orient="records")
    
    # Integration test endpoints
    @app.post("/integration", response_model=SuccessResponse)
    def store_integration(result: IntegrationTestResult):
        result_id = api.store_integration_test_result(result)
        return SuccessResponse(success=True, message="Integration test result stored successfully", result_id=result_id)
    
    @app.get("/integration")
    def get_integration(
        test_module: Optional[str] = None
    ):
        df = api.get_integration_test_summary(test_module)
        return df.to_dict(orient="records")
    
    # Utility endpoints
    @app.post("/query")
    def execute_query(query_request: QueryRequest):
        df = api.query(query_request.sql, query_request.parameters)
        return df.to_dict(orient="records")
    
    @app.get("/hardware")
    def get_hardware():
        df = api.get_hardware_list()
        return df.to_dict(orient="records")
    
    @app.get("/models")
    def get_models():
        df = api.get_model_list()
        return df.to_dict(orient="records")
    
    return app

def main():
    """Command-line interface for the benchmark database API."""
    parser = argparse.ArgumentParser(description="Benchmark Database API")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--serve", action="store_true",
                       help="Start the API server")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind the API server to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind the API server to")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    if args.serve:
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()