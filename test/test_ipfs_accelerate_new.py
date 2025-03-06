#\!/usr/bin/env python
"""
IPFS Accelerate Python Test Framework

This script provides comprehensive testing for IPFS acceleration across different hardware platforms,
with integrated DuckDB support for test result storage and analysis.

Key features:
- Tests IPFS acceleration on various hardware platforms (CPU, CUDA, OpenVINO, QNN, WebNN, WebGPU)
- Measures performance metrics including latency, throughput, and power consumption
- Stores test results in DuckDB database for efficient querying and analysis
- Generates comprehensive reports in multiple formats (markdown, HTML, JSON)
- Supports P2P network optimization tests for content distribution
- Includes battery impact analysis for mobile/edge devices

Usage examples:
  python test_ipfs_accelerate.py --models "bert-base-uncased" --db-only
  python test_ipfs_accelerate.py --comparison-report --format html
  python test_ipfs_accelerate.py --webgpu-analysis --browser firefox --format html
  python test_ipfs_accelerate.py --models "bert-base-uncased" --p2p-optimization
"""

import asyncio
import os
import sys
import json
import time
import traceback
import argparse
import platform
import multiprocessing
from pathlib import Path
from datetime import datetime
import importlib.util
from typing import Dict, List, Any, Optional, Union
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

# Set environment variables to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Determine if JSON output should be deprecated in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

# Set environment variable to avoid fork warnings in multiprocessing
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

# Configure to use spawn instead of fork to prevent deadlocks
if hasattr(multiprocessing, "set_start_method"):
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Could not set multiprocessing start method to 'spawn' - already set")

# Add parent directory to sys.path for proper imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import DuckDB and related dependencies
try:
    import duckdb
    import pandas as pd
    HAVE_DUCKDB = True
    print("DuckDB support enabled for test results")
except ImportError:
    HAVE_DUCKDB = False
    if DEPRECATE_JSON_OUTPUT:
        print("Warning: DuckDB not installed but DEPRECATE_JSON_OUTPUT=1. Will still save JSON as fallback.")
        print("To enable database storage, install duckdb: pip install duckdb pandas")

# Try to import Plotly for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
    print("Plotly visualization support enabled")
except ImportError:
    HAVE_PLOTLY = False
    print("Plotly not installed. Interactive visualizations will be disabled.")
    print("To enable interactive visualizations, install plotly: pip install plotly")

class TestResultsDBHandler:
    """
    Handler for storing test results in DuckDB database.
    This class abstracts away the database operations to store test results.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the database handler.
        
        Args:
            db_path: Path to DuckDB database file. If None, uses BENCHMARK_DB_PATH
                    environment variable or default path ./benchmark_db.duckdb
        """
        # Skip initialization if DuckDB is not available
        if not HAVE_DUCKDB:
            self.db_path = None
            self.con = None
            print("DuckDB not available - results will not be stored in database")
            return
            
        # Get database path from environment or argument
        if db_path is None:
            self.db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        else:
            self.db_path = db_path
            
        try:
            # Connect to DuckDB database directly
            self.con = duckdb.connect(self.db_path)
            print(f"Connected to DuckDB database at: {self.db_path}")
            
            # Create necessary tables
            self._create_tables()
        except Exception as e:
            print(f"Warning: Failed to initialize database connection: {e}")
            self.con = None
            
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        if self.con is None:
            return
            
        try:
            # Create hardware_platforms table
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS hardware_platforms (
                    hardware_id INTEGER PRIMARY KEY,
                    hardware_type VARCHAR,
                    device_name VARCHAR,
                    compute_units INTEGER,
                    memory_capacity FLOAT,
                    driver_version VARCHAR,
                    supported_precisions VARCHAR,
                    max_batch_size INTEGER,
                    detected_at TIMESTAMP
                )
            """)
            
            # Create models table
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    model_family VARCHAR,
                    model_type VARCHAR,
                    model_size VARCHAR,
                    parameters_million FLOAT,
                    added_at TIMESTAMP
                )
            """)
            
            # Create test_results table
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    test_date VARCHAR,
                    status VARCHAR,
                    test_type VARCHAR,
                    model_id INTEGER,
                    hardware_id INTEGER,
                    endpoint_type VARCHAR,
                    success BOOLEAN,
                    error_message VARCHAR,
                    execution_time FLOAT,
                    memory_usage FLOAT,
                    details VARCHAR,
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
                )
            """)
            
            # Create performance_results table
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS performance_results (
                    id INTEGER PRIMARY KEY,
                    model_id INTEGER,
                    hardware_id INTEGER,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    average_latency_ms FLOAT,
                    p50_latency_ms FLOAT,
                    p90_latency_ms FLOAT,
                    p99_latency_ms FLOAT,
                    throughput_items_per_second FLOAT,
                    memory_peak_mb FLOAT,
                    power_watts FLOAT,
                    energy_efficiency_items_per_joule FLOAT,
                    test_timestamp TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
                )
            """)
            
            # Create hardware_compatibility table
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS hardware_compatibility (
                    id INTEGER PRIMARY KEY,
                    model_id INTEGER,
                    hardware_id INTEGER,
                    compatibility_status VARCHAR,
                    compatibility_score FLOAT,
                    recommended BOOLEAN,
                    last_tested TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
                )
            """)
            
            # Create power_metrics table for mobile/edge devices
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS power_metrics (
                    id INTEGER PRIMARY KEY,
                    test_id INTEGER,
                    model_id INTEGER,
                    hardware_id INTEGER,
                    power_watts_avg FLOAT,
                    power_watts_peak FLOAT,
                    temperature_celsius_avg FLOAT,
                    temperature_celsius_peak FLOAT,
                    battery_impact_mah FLOAT,
                    test_duration_seconds FLOAT,
                    estimated_runtime_hours FLOAT,
                    test_timestamp TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES test_results(id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
                )
            """)
            
            # Create ipfs_acceleration_results table for IPFS-specific metrics
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS ipfs_acceleration_results (
                    id INTEGER PRIMARY KEY,
                    test_id INTEGER,
                    model_id INTEGER,
                    cid VARCHAR,
                    source VARCHAR,
                    transfer_time_ms FLOAT,
                    p2p_optimized BOOLEAN,
                    peer_count INTEGER,
                    network_efficiency FLOAT,
                    optimization_score FLOAT,
                    load_time_ms FLOAT,
                    test_timestamp TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES test_results(id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)
            
            # Create p2p_network_metrics table for P2P specific metrics
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS p2p_network_metrics (
                    id INTEGER PRIMARY KEY,
                    ipfs_result_id INTEGER,
                    peer_count INTEGER,
                    known_content_items INTEGER,
                    transfers_completed INTEGER,
                    transfers_failed INTEGER,
                    bytes_transferred BIGINT,
                    average_transfer_speed FLOAT,
                    network_efficiency FLOAT,
                    network_density FLOAT,
                    average_connections FLOAT,
                    optimization_score FLOAT,
                    optimization_rating VARCHAR,
                    network_health VARCHAR,
                    test_timestamp TIMESTAMP,
                    FOREIGN KEY (ipfs_result_id) REFERENCES ipfs_acceleration_results(id)
                )
            """)
            
            # Create webgpu_metrics table for WebGPU specific metrics
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS webgpu_metrics (
                    id INTEGER PRIMARY KEY,
                    test_id INTEGER,
                    browser_name VARCHAR,
                    browser_version VARCHAR,
                    compute_shaders_enabled BOOLEAN,
                    shader_precompilation_enabled BOOLEAN,
                    parallel_loading_enabled BOOLEAN,
                    shader_compile_time_ms FLOAT,
                    first_inference_time_ms FLOAT,
                    subsequent_inference_time_ms FLOAT,
                    pipeline_creation_time_ms FLOAT,
                    workgroup_size VARCHAR,
                    optimization_score FLOAT,
                    test_timestamp TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES test_results(id)
                )
            """)
            
            print("Database tables created successfully")
        except Exception as e:
            print(f"Error creating database tables: {e}")
            traceback.print_exc()
            
    def _get_or_create_model(self, model_name, model_family=None, model_type=None, model_size=None, parameters_million=None):
        """Get model ID from database or create new entry if it doesn't exist."""
        if self.con is None or not model_name:
            return None
            
        try:
            # Check if model exists
            result = self.con.execute(
                "SELECT model_id FROM models WHERE model_name = ?", 
                [model_name]
            ).fetchone()
            
            if result:
                return result[0]
                
            # Create new model entry
            now = datetime.now()
            self.con.execute(
                """
                INSERT INTO models (model_name, model_family, model_type, model_size, parameters_million, added_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [model_name, model_family, model_type, model_size, parameters_million, now]
            )
            
            # Get the newly created ID
            result = self.con.execute(
                "SELECT model_id FROM models WHERE model_name = ?", 
                [model_name]
            ).fetchone()
            
            return result[0] if result else None
        except Exception as e:
            print(f"Error in _get_or_create_model: {e}")
            return None
            
    def _get_or_create_hardware(self, hardware_type, device_name=None, compute_units=None, 
                               memory_capacity=None, driver_version=None, supported_precisions=None,
                               max_batch_size=None):
        """Get hardware ID from database or create new entry if it doesn't exist."""
        if self.con is None or not hardware_type:
            return None
            
        try:
            # Check if hardware platform exists
            result = self.con.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND (device_name = ? OR (device_name IS NULL AND ? IS NULL))", 
                [hardware_type, device_name, device_name]
            ).fetchone()
            
            if result:
                return result[0]
                
            # Create new hardware platform entry
            now = datetime.now()
            self.con.execute(
                """
                INSERT INTO hardware_platforms (
                    hardware_type, device_name, compute_units, memory_capacity, 
                    driver_version, supported_precisions, max_batch_size, detected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [hardware_type, device_name, compute_units, memory_capacity,
                 driver_version, supported_precisions, max_batch_size, now]
            )
            
            # Get the newly created ID
            result = self.con.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND (device_name = ? OR (device_name IS NULL AND ? IS NULL))", 
                [hardware_type, device_name, device_name]
            ).fetchone()
            
            return result[0] if result else None
        except Exception as e:
            print(f"Error in _get_or_create_hardware: {e}")
            return None
            
    def store_test_result(self, test_result):
        """Store a test result in the database."""
        if self.con is None or not test_result:
            return False
            
        try:
            # Extract values from test_result
            model_name = test_result.get('model_name')
            model_family = test_result.get('model_family')
            hardware_type = test_result.get('hardware_type')
            
            # Get or create model and hardware entries
            model_id = self._get_or_create_model(model_name, model_family)
            hardware_id = self._get_or_create_hardware(hardware_type)
            
            if not model_id or not hardware_id:
                print(f"Warning: Could not get/create model or hardware ID for {model_name} on {hardware_type}")
                return False
                
            # Prepare test data
            now = datetime.now()
            test_date = now.strftime("%Y-%m-%d")
            
            # Store main test result
            self.con.execute(
                """
                INSERT INTO test_results (
                    timestamp, test_date, status, test_type, model_id, hardware_id,
                    endpoint_type, success, error_message, execution_time, memory_usage, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    now, test_date, 
                    test_result.get('status'),
                    test_result.get('test_type'),
                    model_id, hardware_id,
                    test_result.get('endpoint_type'),
                    test_result.get('success', False),
                    test_result.get('error_message'),
                    test_result.get('execution_time'),
                    test_result.get('memory_usage'),
                    json.dumps(test_result.get('details', {}))
                ]
            )
            
            # Get the newly created test result ID
            result = self.con.execute(
                """
                SELECT id FROM test_results 
                WHERE model_id = ? AND hardware_id = ? 
                ORDER BY timestamp DESC LIMIT 1
                """, 
                [model_id, hardware_id]
            ).fetchone()
            
            test_id = result[0] if result else None
            
            # Store performance metrics if available
            if test_id and 'performance' in test_result:
                self._store_performance_metrics(test_id, model_id, hardware_id, test_result['performance'])
                
            # Store power metrics if available
            if test_id and 'power_metrics' in test_result:
                self._store_power_metrics(test_id, model_id, hardware_id, test_result['power_metrics'])
                
            # Store hardware compatibility if available
            if 'compatibility' in test_result:
                self._store_hardware_compatibility(model_id, hardware_id, test_result['compatibility'])
                
            # Store IPFS acceleration results if available
            if test_id and 'ipfs_acceleration' in test_result:
                self._store_ipfs_acceleration_results(test_id, model_id, test_result['ipfs_acceleration'])
                
            # Store WebGPU metrics if available
            if test_id and 'webgpu_metrics' in test_result:
                self._store_webgpu_metrics(test_id, test_result['webgpu_metrics'])
                
            return True
        except Exception as e:
            print(f"Error storing test result: {e}")
            traceback.print_exc()
            return False
            
    def _store_performance_metrics(self, test_id, model_id, hardware_id, performance):
        """Store performance metrics in the database."""
        if self.con is None or not performance:
            return False
            
        try:
            now = datetime.now()
            self.con.execute(
                """
                INSERT INTO performance_results (
                    model_id, hardware_id, batch_size, sequence_length,
                    average_latency_ms, p50_latency_ms, p90_latency_ms, p99_latency_ms,
                    throughput_items_per_second, memory_peak_mb, power_watts,
                    energy_efficiency_items_per_joule, test_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    model_id, hardware_id,
                    performance.get('batch_size'),
                    performance.get('sequence_length'),
                    performance.get('average_latency_ms'),
                    performance.get('p50_latency_ms'),
                    performance.get('p90_latency_ms'),
                    performance.get('p99_latency_ms'),
                    performance.get('throughput_items_per_second'),
                    performance.get('memory_peak_mb'),
                    performance.get('power_watts'),
                    performance.get('energy_efficiency_items_per_joule'),
                    now
                ]
            )
            return True
        except Exception as e:
            print(f"Error storing performance metrics: {e}")
            return False
            
    def _store_power_metrics(self, test_id, model_id, hardware_id, power_metrics):
        """Store power metrics in the database."""
        if self.con is None or not power_metrics:
            return False
            
        try:
            now = datetime.now()
            self.con.execute(
                """
                INSERT INTO power_metrics (
                    test_id, model_id, hardware_id, power_watts_avg, power_watts_peak,
                    temperature_celsius_avg, temperature_celsius_peak, battery_impact_mah,
                    test_duration_seconds, estimated_runtime_hours, test_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    test_id, model_id, hardware_id,
                    power_metrics.get('power_watts_avg'),
                    power_metrics.get('power_watts_peak'),
                    power_metrics.get('temperature_celsius_avg'),
                    power_metrics.get('temperature_celsius_peak'),
                    power_metrics.get('battery_impact_mah'),
                    power_metrics.get('test_duration_seconds'),
                    power_metrics.get('estimated_runtime_hours'),
                    now
                ]
            )
            return True
        except Exception as e:
            print(f"Error storing power metrics: {e}")
            return False
            
    def _store_hardware_compatibility(self, model_id, hardware_id, compatibility):
        """Store hardware compatibility information in the database."""
        if self.con is None or not compatibility:
            return False
            
        try:
            now = datetime.now()
            
            # Check if compatibility entry exists
            result = self.con.execute(
                """
                SELECT id FROM hardware_compatibility 
                WHERE model_id = ? AND hardware_id = ?
                """, 
                [model_id, hardware_id]
            ).fetchone()
            
            if result:
                # Update existing entry
                self.con.execute(
                    """
                    UPDATE hardware_compatibility SET
                    compatibility_status = ?,
                    compatibility_score = ?,
                    recommended = ?,
                    last_tested = ?
                    WHERE model_id = ? AND hardware_id = ?
                    """,
                    [
                        compatibility.get('status'),
                        compatibility.get('score'),
                        compatibility.get('recommended', False),
                        now,
                        model_id, hardware_id
                    ]
                )
            else:
                # Create new entry
                self.con.execute(
                    """
                    INSERT INTO hardware_compatibility (
                        model_id, hardware_id, compatibility_status,
                        compatibility_score, recommended, last_tested
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        model_id, hardware_id,
                        compatibility.get('status'),
                        compatibility.get('score'),
                        compatibility.get('recommended', False),
                        now
                    ]
                )
            return True
        except Exception as e:
            print(f"Error storing hardware compatibility: {e}")
            return False
            
    def _store_ipfs_acceleration_results(self, test_id, model_id, ipfs_acceleration):
        """Store IPFS acceleration results in the database."""
        if self.con is None or not ipfs_acceleration:
            return False
            
        try:
            now = datetime.now()
            
            # Insert IPFS acceleration result
            self.con.execute(
                """
                INSERT INTO ipfs_acceleration_results (
                    test_id, model_id, cid, source, transfer_time_ms,
                    p2p_optimized, peer_count, network_efficiency,
                    optimization_score, load_time_ms, test_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    test_id, model_id,
                    ipfs_acceleration.get('cid'),
                    ipfs_acceleration.get('source'),
                    ipfs_acceleration.get('transfer_time_ms'),
                    ipfs_acceleration.get('p2p_optimized', False),
                    ipfs_acceleration.get('peer_count'),
                    ipfs_acceleration.get('network_efficiency'),
                    ipfs_acceleration.get('optimization_score'),
                    ipfs_acceleration.get('load_time_ms'),
                    now
                ]
            )
            
            # Get the newly created IPFS result ID
            result = self.con.execute(
                """
                SELECT id FROM ipfs_acceleration_results 
                WHERE test_id = ? 
                ORDER BY test_timestamp DESC LIMIT 1
                """, 
                [test_id]
            ).fetchone()
            
            ipfs_result_id = result[0] if result else None
            
            # Store P2P network metrics if available
            if ipfs_result_id and 'p2p_metrics' in ipfs_acceleration:
                self._store_p2p_network_metrics(ipfs_result_id, ipfs_acceleration['p2p_metrics'])
                
            return True
        except Exception as e:
            print(f"Error storing IPFS acceleration results: {e}")
            return False
            
    def _store_p2p_network_metrics(self, ipfs_result_id, p2p_metrics):
        """Store P2P network metrics in the database."""
        if self.con is None or not p2p_metrics:
            return False
            
        try:
            now = datetime.now()
            
            self.con.execute(
                """
                INSERT INTO p2p_network_metrics (
                    ipfs_result_id, peer_count, known_content_items, transfers_completed,
                    transfers_failed, bytes_transferred, average_transfer_speed,
                    network_efficiency, network_density, average_connections,
                    optimization_score, optimization_rating, network_health,
                    test_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ipfs_result_id,
                    p2p_metrics.get('peer_count'),
                    p2p_metrics.get('known_content_items'),
                    p2p_metrics.get('transfers_completed'),
                    p2p_metrics.get('transfers_failed'),
                    p2p_metrics.get('bytes_transferred'),
                    p2p_metrics.get('average_transfer_speed'),
                    p2p_metrics.get('network_efficiency'),
                    p2p_metrics.get('network_density'),
                    p2p_metrics.get('average_connections'),
                    p2p_metrics.get('optimization_score'),
                    p2p_metrics.get('optimization_rating'),
                    p2p_metrics.get('network_health'),
                    now
                ]
            )
            return True
        except Exception as e:
            print(f"Error storing P2P network metrics: {e}")
            return False
            
    def _store_webgpu_metrics(self, test_id, webgpu_metrics):
        """Store WebGPU metrics in the database."""
        if self.con is None or not webgpu_metrics:
            return False
            
        try:
            now = datetime.now()
            
            self.con.execute(
                """
                INSERT INTO webgpu_metrics (
                    test_id, browser_name, browser_version, compute_shaders_enabled,
                    shader_precompilation_enabled, parallel_loading_enabled,
                    shader_compile_time_ms, first_inference_time_ms,
                    subsequent_inference_time_ms, pipeline_creation_time_ms,
                    workgroup_size, optimization_score, test_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    test_id,
                    webgpu_metrics.get('browser_name'),
                    webgpu_metrics.get('browser_version'),
                    webgpu_metrics.get('compute_shaders_enabled', False),
                    webgpu_metrics.get('shader_precompilation_enabled', False),
                    webgpu_metrics.get('parallel_loading_enabled', False),
                    webgpu_metrics.get('shader_compile_time_ms'),
                    webgpu_metrics.get('first_inference_time_ms'),
                    webgpu_metrics.get('subsequent_inference_time_ms'),
                    webgpu_metrics.get('pipeline_creation_time_ms'),
                    webgpu_metrics.get('workgroup_size'),
                    webgpu_metrics.get('optimization_score'),
                    now
                ]
            )
            return True
        except Exception as e:
            print(f"Error storing WebGPU metrics: {e}")
            return False
            
    def query_test_results(self, query, params=None):
        """
        Execute a custom SQL query against the database.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            Pandas DataFrame with query results
        """
        if self.con is None:
            print("Database connection not available")
            return None
            
        try:
            if params:
                result = self.con.execute(query, params).fetchdf()
            else:
                result = self.con.execute(query).fetchdf()
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
            
    def create_hardware_compatibility_matrix(self, model_types=None):
        """
        Create a hardware compatibility matrix from the database.
        
        Args:
            model_types: Optional list of model types to include in the matrix
            
        Returns:
            Pandas DataFrame with compatibility matrix
        """
        if self.con is None:
            print("Database connection not available")
            return None
            
        try:
            # Build the query
            query = """
            SELECT 
                m.model_name,
                m.model_family,
                m.model_type,
                m.model_size,
                hp.hardware_type,
                hc.compatibility_status,
                hc.compatibility_score,
                hc.recommended,
                hc.last_tested
            FROM hardware_compatibility hc
            JOIN models m ON hc.model_id = m.model_id
            JOIN hardware_platforms hp ON hc.hardware_id = hp.hardware_id
            """
            
            # Add filter for model types if provided
            if model_types:
                placeholders = ", ".join(["?"] * len(model_types))
                query += f" WHERE m.model_type IN ({placeholders})"
                result = self.con.execute(query, model_types).fetchdf()
            else:
                result = self.con.execute(query).fetchdf()
                
            return result
        except Exception as e:
            print(f"Error creating hardware compatibility matrix: {e}")
            return None
            
    def get_ipfs_acceleration_results(self, model_name=None, limit=100):
        """
        Get IPFS acceleration results from the database.
        
        Args:
            model_name: Optional model name to filter results
            limit: Maximum number of results to return
            
        Returns:
            Pandas DataFrame with IPFS acceleration results
        """
        if self.con is None:
            print("Database connection not available")
            return None
            
        try:
            # Build the query
            query = """
            SELECT 
                m.model_name,
                m.model_family,
                m.model_type,
                hp.hardware_type,
                iar.cid,
                iar.source,
                iar.transfer_time_ms,
                iar.p2p_optimized,
                iar.peer_count,
                iar.network_efficiency,
                iar.optimization_score,
                iar.load_time_ms,
                iar.test_timestamp
            FROM ipfs_acceleration_results iar
            JOIN models m ON iar.model_id = m.model_id
            JOIN test_results tr ON iar.test_id = tr.id
            JOIN hardware_platforms hp ON tr.hardware_id = hp.hardware_id
            """
            
            # Add filter for model name if provided
            if model_name:
                query += " WHERE m.model_name = ?"
                query += f" ORDER BY iar.test_timestamp DESC LIMIT {limit}"
                result = self.con.execute(query, [model_name]).fetchdf()
            else:
                query += f" ORDER BY iar.test_timestamp DESC LIMIT {limit}"
                result = self.con.execute(query).fetchdf()
                
            return result
        except Exception as e:
            print(f"Error getting IPFS acceleration results: {e}")
            return None
            
    def get_p2p_network_metrics(self, limit=100):
        """
        Get P2P network metrics from the database.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            Pandas DataFrame with P2P network metrics
        """
        if self.con is None:
            print("Database connection not available")
            return None
            
        try:
            query = """
            SELECT 
                m.model_name,
                hp.hardware_type,
                iar.cid,
                iar.source,
                iar.p2p_optimized,
                pnm.peer_count,
                pnm.known_content_items,
                pnm.transfers_completed,
                pnm.transfers_failed,
                pnm.bytes_transferred,
                pnm.average_transfer_speed,
                pnm.network_efficiency,
                pnm.network_density,
                pnm.average_connections,
                pnm.optimization_score,
                pnm.optimization_rating,
                pnm.network_health,
                pnm.test_timestamp
            FROM p2p_network_metrics pnm
            JOIN ipfs_acceleration_results iar ON pnm.ipfs_result_id = iar.id
            JOIN models m ON iar.model_id = m.model_id
            JOIN test_results tr ON iar.test_id = tr.id
            JOIN hardware_platforms hp ON tr.hardware_id = hp.hardware_id
            ORDER BY pnm.test_timestamp DESC
            LIMIT ?
            """
            
            result = self.con.execute(query, [limit]).fetchdf()
            return result
        except Exception as e:
            print(f"Error getting P2P network metrics: {e}")
            return None
            
    def get_webgpu_metrics(self, browser_name=None, limit=100):
        """
        Get WebGPU metrics from the database.
        
        Args:
            browser_name: Optional browser name to filter results
            limit: Maximum number of results to return
            
        Returns:
            Pandas DataFrame with WebGPU metrics
        """
        if self.con is None:
            print("Database connection not available")
            return None
            
        try:
            # Build the query
            query = """
            SELECT 
                m.model_name,
                m.model_type,
                hp.hardware_type,
                wm.browser_name,
                wm.browser_version,
                wm.compute_shaders_enabled,
                wm.shader_precompilation_enabled,
                wm.parallel_loading_enabled,
                wm.shader_compile_time_ms,
                wm.first_inference_time_ms,
                wm.subsequent_inference_time_ms,
                wm.pipeline_creation_time_ms,
                wm.workgroup_size,
                wm.optimization_score,
                wm.test_timestamp
            FROM webgpu_metrics wm
            JOIN test_results tr ON wm.test_id = tr.id
            JOIN models m ON tr.model_id = m.model_id
            JOIN hardware_platforms hp ON tr.hardware_id = hp.hardware_id
            """
            
            # Add filter for browser name if provided
            if browser_name:
                query += " WHERE wm.browser_name = ?"
                query += f" ORDER BY wm.test_timestamp DESC LIMIT {limit}"
                result = self.con.execute(query, [browser_name]).fetchdf()
            else:
                query += f" ORDER BY wm.test_timestamp DESC LIMIT {limit}"
                result = self.con.execute(query).fetchdf()
                
            return result
        except Exception as e:
            print(f"Error getting WebGPU metrics: {e}")
            return None
    
    def generate_report(self, report_type, format="markdown", output=None):
        """
        Generate a report from test results in the database.
        
        Args:
            report_type: Type of report to generate (performance, compatibility, ipfs, webgpu)
            format: Output format (markdown, html, json)
            output: Optional output file path
            
        Returns:
            Report content as string
        """
        if self.con is None:
            print("Database connection not available")
            return "Database connection not available"
            
        try:
            if report_type == "performance":
                return self._generate_performance_report(format, output)
            elif report_type == "compatibility":
                return self._generate_compatibility_report(format, output)
            elif report_type == "ipfs":
                return self._generate_ipfs_report(format, output)
            elif report_type == "webgpu":
                return self._generate_webgpu_report(format, output)
            elif report_type == "p2p":
                return self._generate_p2p_report(format, output)
            else:
                return f"Unknown report type: {report_type}"
        except Exception as e:
            print(f"Error generating report: {e}")
            traceback.print_exc()
            return f"Error generating report: {e}"
            
    def _generate_performance_report(self, format="markdown", output=None):
        """Generate a performance report."""
        try:
            # Get performance data
            query = """
            SELECT 
                m.model_name,
                m.model_family,
                m.model_type,
                hp.hardware_type,
                pr.batch_size,
                pr.sequence_length,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                pr.power_watts,
                pr.energy_efficiency_items_per_joule,
                pr.test_timestamp
            FROM performance_results pr
            JOIN models m ON pr.model_id = m.model_id
            JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            ORDER BY pr.test_timestamp DESC
            LIMIT 1000
            """
            
            df = self.con.execute(query).fetchdf()
            
            if df.empty:
                return "No performance data available"
                
            if format == "json":
                # Convert to JSON
                result = df.to_json(orient="records", indent=2)
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(result)
                        
                return result
            
            # Group by model type and hardware type
            grouped = df.groupby(["model_type", "hardware_type"]).agg({
                "average_latency_ms": "mean",
                "throughput_items_per_second": "mean",
                "memory_peak_mb": "mean",
                "power_watts": "mean",
                "energy_efficiency_items_per_joule": "mean"
            }).reset_index()
            
            if format == "markdown":
                # Create markdown report
                report = "# Performance Report\n\n"
                report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                report += "## Summary\n\n"
                report += "| Model Type | Hardware | Avg Latency (ms) | Throughput (items/s) | Memory (MB) | Power (W) | Efficiency (items/J) |\n"
                report += "|------------|----------|------------------|----------------------|-------------|-----------|----------------------|\n"
                
                for _, row in grouped.iterrows():
                    report += f"| {row['model_type']} | {row['hardware_type']} | {row['average_latency_ms']:.2f} | {row['throughput_items_per_second']:.2f} | {row['memory_peak_mb']:.2f} | {row['power_watts']:.2f} | {row['energy_efficiency_items_per_joule']:.2f} |\n"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            elif format == "html":
                # Create HTML report with Plotly visualizations
                if not HAVE_PLOTLY:
                    return "Plotly not installed. Cannot generate HTML report."
                    
                # Create figures
                fig1 = px.bar(
                    grouped, 
                    x="model_type", 
                    y="throughput_items_per_second", 
                    color="hardware_type",
                    title="Throughput by Model Type and Hardware",
                    labels={"throughput_items_per_second": "Throughput (items/s)", "model_type": "Model Type", "hardware_type": "Hardware"}
                )
                
                fig2 = px.scatter(
                    grouped,
                    x="average_latency_ms",
                    y="throughput_items_per_second",
                    size="memory_peak_mb",
                    color="hardware_type",
                    hover_name="model_type",
                    title="Latency vs Throughput by Hardware",
                    labels={"average_latency_ms": "Average Latency (ms)", "throughput_items_per_second": "Throughput (items/s)", "memory_peak_mb": "Memory (MB)", "hardware_type": "Hardware"}
                )
                
                # Combine figures
                report = "<html><head><title>Performance Report</title></head><body>"
                report += f"<h1>Performance Report</h1>"
                report += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
                
                report += f"<div>{fig1.to_html(full_html=False)}</div>"
                report += f"<div>{fig2.to_html(full_html=False)}</div>"
                
                report += "</body></html>"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            else:
                return f"Unsupported format: {format}"
        
        except Exception as e:
            print(f"Error generating performance report: {e}")
            traceback.print_exc()
            return f"Error generating performance report: {e}"
            
    def _generate_compatibility_report(self, format="markdown", output=None):
        """Generate a hardware compatibility report."""
        try:
            # Get compatibility data
            query = """
            SELECT 
                m.model_name,
                m.model_family,
                m.model_type,
                hp.hardware_type,
                hc.compatibility_status,
                hc.compatibility_score,
                hc.recommended,
                hc.last_tested
            FROM hardware_compatibility hc
            JOIN models m ON hc.model_id = m.model_id
            JOIN hardware_platforms hp ON hc.hardware_id = hp.hardware_id
            ORDER BY m.model_type, m.model_name, hp.hardware_type
            """
            
            df = self.con.execute(query).fetchdf()
            
            if df.empty:
                return "No compatibility data available"
                
            if format == "json":
                # Convert to JSON
                result = df.to_json(orient="records", indent=2)
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(result)
                        
                return result
            
            # Create a pivot table for compatibility
            pivot = df.pivot_table(
                index=["model_type", "model_name"],
                columns="hardware_type",
                values="compatibility_status",
                aggfunc="first"
            ).reset_index()
            
            if format == "markdown":
                # Create markdown report
                report = "# Hardware Compatibility Matrix\n\n"
                report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # Group by model type
                model_types = df["model_type"].unique()
                
                for model_type in model_types:
                    report += f"## {model_type} Models\n\n"
                    
                    type_df = pivot[pivot["model_type"] == model_type]
                    
                    # Create table header
                    hardware_cols = [col for col in type_df.columns if col not in ["model_type", "model_name"]]
                    header = "| Model | " + " | ".join(hardware_cols) + " |\n"
                    separator = "|-------|" + "|".join(["-" * len(col) for col in hardware_cols]) + "|\n"
                    
                    report += header
                    report += separator
                    
                    # Add rows
                    for _, row in type_df.iterrows():
                        model_name = row["model_name"]
                        row_values = []
                        
                        for hw in hardware_cols:
                            status = row.get(hw, "")
                            
                            if pd.isna(status):
                                cell = "❓"
                            elif status == "compatible":
                                cell = "✅"
                            elif status == "limited":
                                cell = "⚠️"
                            elif status == "incompatible":
                                cell = "❌"
                            else:
                                cell = "❓"
                                
                            row_values.append(cell)
                            
                        report += f"| {model_name} | " + " | ".join(row_values) + " |\n"
                        
                    report += "\n"
                
                # Add legend
                report += "## Legend\n\n"
                report += "- ✅ Compatible: Fully supported\n"
                report += "- ⚠️ Limited: Supported with limitations\n"
                report += "- ❌ Incompatible: Not supported\n"
                report += "- ❓ Unknown: Not tested\n"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            elif format == "html":
                # Create HTML report with interactive heatmap
                if not HAVE_PLOTLY:
                    return "Plotly not installed. Cannot generate HTML report."
                
                # Convert pivot table to a format suitable for heatmap
                pivot_flat = []
                
                for _, row in pivot.iterrows():
                    model_type = row["model_type"]
                    model_name = row["model_name"]
                    
                    for hw in [col for col in row.index if col not in ["model_type", "model_name"]]:
                        status = row[hw]
                        
                        if pd.isna(status):
                            score = 0  # Unknown
                        elif status == "compatible":
                            score = 1  # Compatible
                        elif status == "limited":
                            score = 0.5  # Limited
                        elif status == "incompatible":
                            score = 0  # Incompatible
                        else:
                            score = 0  # Unknown
                            
                        pivot_flat.append({
                            "model_type": model_type,
                            "model_name": model_name,
                            "hardware_type": hw,
                            "compatibility_score": score
                        })
                
                heatmap_df = pd.DataFrame(pivot_flat)
                
                # Create heatmap figure
                fig = px.density_heatmap(
                    heatmap_df,
                    x="hardware_type",
                    y="model_name",
                    z="compatibility_score",
                    color_continuous_scale=[(0, "red"), (0.5, "yellow"), (1, "green")],
                    labels={"hardware_type": "Hardware Type", "model_name": "Model", "compatibility_score": "Compatibility"},
                    title="Hardware Compatibility Matrix",
                    facet_row="model_type"
                )
                
                fig.update_layout(height=800)
                
                # Create HTML report
                report = "<html><head><title>Hardware Compatibility Matrix</title></head><body>"
                report += f"<h1>Hardware Compatibility Matrix</h1>"
                report += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
                
                report += f"<div>{fig.to_html(full_html=False)}</div>"
                
                report += "<h2>Legend</h2>"
                report += "<ul>"
                report += "<li><span style='color:green'>■</span> Compatible: Fully supported</li>"
                report += "<li><span style='color:yellow'>■</span> Limited: Supported with limitations</li>"
                report += "<li><span style='color:red'>■</span> Incompatible: Not supported</li>"
                report += "<li><span style='color:lightgray'>■</span> Unknown: Not tested</li>"
                report += "</ul>"
                
                report += "</body></html>"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            else:
                return f"Unsupported format: {format}"
        
        except Exception as e:
            print(f"Error generating compatibility report: {e}")
            traceback.print_exc()
            return f"Error generating compatibility report: {e}"
    
    def _generate_ipfs_report(self, format="markdown", output=None):
        """Generate an IPFS acceleration report."""
        try:
            # Get IPFS acceleration data
            query = """
            SELECT 
                m.model_name,
                m.model_type,
                hp.hardware_type,
                iar.cid,
                iar.source,
                iar.transfer_time_ms,
                iar.p2p_optimized,
                iar.peer_count,
                iar.network_efficiency,
                iar.optimization_score,
                iar.load_time_ms,
                iar.test_timestamp
            FROM ipfs_acceleration_results iar
            JOIN models m ON iar.model_id = m.model_id
            JOIN test_results tr ON iar.test_id = tr.id
            JOIN hardware_platforms hp ON tr.hardware_id = hp.hardware_id
            ORDER BY iar.test_timestamp DESC
            LIMIT 100
            """
            
            df = self.con.execute(query).fetchdf()
            
            if df.empty:
                return "No IPFS acceleration data available"
                
            if format == "json":
                # Convert to JSON
                result = df.to_json(orient="records", indent=2)
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(result)
                        
                return result
            
            # Compare P2P vs standard IPFS
            p2p_df = df[df["p2p_optimized"] == True]
            std_df = df[df["p2p_optimized"] == False]
            
            p2p_avg_transfer = p2p_df["transfer_time_ms"].mean() if not p2p_df.empty else 0
            std_avg_transfer = std_df["transfer_time_ms"].mean() if not std_df.empty else 0
            
            p2p_avg_load = p2p_df["load_time_ms"].mean() if not p2p_df.empty else 0
            std_avg_load = std_df["load_time_ms"].mean() if not std_df.empty else 0
            
            if p2p_avg_transfer > 0 and std_avg_transfer > 0:
                improvement_pct = ((std_avg_transfer - p2p_avg_transfer) / std_avg_transfer) * 100
            else:
                improvement_pct = 0
                
            # Calculate average optimization scores
            avg_opt_score = df["optimization_score"].mean() if "optimization_score" in df.columns else 0
            avg_network_efficiency = df["network_efficiency"].mean() if "network_efficiency" in df.columns else 0
            
            if format == "markdown":
                # Create markdown report
                report = "# IPFS Acceleration Report\n\n"
                report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                report += "## Summary\n\n"
                report += f"- Total tests: {len(df)}\n"
                report += f"- P2P optimized tests: {len(p2p_df)}\n"
                report += f"- Standard IPFS tests: {len(std_df)}\n"
                report += f"- Average optimization score: {avg_opt_score:.2f}\n"
                report += f"- Average network efficiency: {avg_network_efficiency:.2f}\n\n"
                
                report += "## Performance Comparison\n\n"
                report += f"- P2P optimized average transfer time: {p2p_avg_transfer:.2f} ms\n"
                report += f"- Standard IPFS average transfer time: {std_avg_transfer:.2f} ms\n"
                report += f"- P2P optimized average load time: {p2p_avg_load:.2f} ms\n"
                report += f"- Standard IPFS average load time: {std_avg_load:.2f} ms\n"
                report += f"- Performance improvement with P2P: {improvement_pct:.2f}%\n\n"
                
                report += "## Recent Tests\n\n"
                report += "| Model | Type | Hardware | Source | Transfer Time (ms) | P2P Optimized | Load Time (ms) | Timestamp |\n"
                report += "|-------|------|----------|--------|-------------------|---------------|----------------|----------|\n"
                
                for _, row in df.head(10).iterrows():
                    p2p = "✓" if row["p2p_optimized"] else "✗"
                    report += f"| {row['model_name']} | {row['model_type']} | {row['hardware_type']} | {row['source']} | {row['transfer_time_ms']:.2f} | {p2p} | {row['load_time_ms']:.2f} | {row['test_timestamp']} |\n"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            elif format == "html":
                # Create HTML report with interactive visualizations
                if not HAVE_PLOTLY:
                    return "Plotly not installed. Cannot generate HTML report."
                
                # Create comparison bar chart
                comparison_data = pd.DataFrame([
                    {"Method": "P2P Optimized", "Time (ms)": p2p_avg_transfer, "Type": "Transfer Time"},
                    {"Method": "Standard IPFS", "Time (ms)": std_avg_transfer, "Type": "Transfer Time"},
                    {"Method": "P2P Optimized", "Time (ms)": p2p_avg_load, "Type": "Load Time"},
                    {"Method": "Standard IPFS", "Time (ms)": std_avg_load, "Type": "Load Time"}
                ])
                
                fig1 = px.bar(
                    comparison_data,
                    x="Method",
                    y="Time (ms)",
                    color="Type",
                    barmode="group",
                    title="P2P vs Standard IPFS Performance",
                    labels={"Method": "Method", "Time (ms)": "Time (ms)"}
                )
                
                # Create performance over time chart
                fig2 = px.scatter(
                    df,
                    x="test_timestamp",
                    y="transfer_time_ms",
                    color="p2p_optimized",
                    size="optimization_score",
                    hover_name="model_name",
                    title="Transfer Time Over Time",
                    labels={"test_timestamp": "Timestamp", "transfer_time_ms": "Transfer Time (ms)", "p2p_optimized": "P2P Optimized"}
                )
                
                # Create HTML report
                report = "<html><head><title>IPFS Acceleration Report</title></head><body>"
                report += f"<h1>IPFS Acceleration Report</h1>"
                report += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
                
                report += "<h2>Summary</h2>"
                report += f"<p>Total tests: {len(df)}</p>"
                report += f"<p>P2P optimized tests: {len(p2p_df)}</p>"
                report += f"<p>Standard IPFS tests: {len(std_df)}</p>"
                report += f"<p>Average optimization score: {avg_opt_score:.2f}</p>"
                report += f"<p>Average network efficiency: {avg_network_efficiency:.2f}</p>"
                
                report += f"<p>Performance improvement with P2P: {improvement_pct:.2f}%</p>"
                
                report += f"<div>{fig1.to_html(full_html=False)}</div>"
                report += f"<div>{fig2.to_html(full_html=False)}</div>"
                
                report += "</body></html>"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            else:
                return f"Unsupported format: {format}"
        
        except Exception as e:
            print(f"Error generating IPFS report: {e}")
            traceback.print_exc()
            return f"Error generating IPFS report: {e}"
            
    def _generate_p2p_report(self, format="markdown", output=None):
        """Generate a P2P network metrics report."""
        try:
            # Get P2P network metrics
            query = """
            SELECT 
                m.model_name,
                m.model_type,
                hp.hardware_type,
                iar.cid,
                iar.source,
                iar.p2p_optimized,
                pnm.peer_count,
                pnm.known_content_items,
                pnm.transfers_completed,
                pnm.transfers_failed,
                pnm.bytes_transferred,
                pnm.average_transfer_speed,
                pnm.network_efficiency,
                pnm.network_density,
                pnm.average_connections,
                pnm.optimization_score,
                pnm.optimization_rating,
                pnm.network_health,
                pnm.test_timestamp
            FROM p2p_network_metrics pnm
            JOIN ipfs_acceleration_results iar ON pnm.ipfs_result_id = iar.id
            JOIN models m ON iar.model_id = m.model_id
            JOIN test_results tr ON iar.test_id = tr.id
            JOIN hardware_platforms hp ON tr.hardware_id = hp.hardware_id
            ORDER BY pnm.test_timestamp DESC
            LIMIT 100
            """
            
            df = self.con.execute(query).fetchdf()
            
            if df.empty:
                return "No P2P network metrics available"
                
            if format == "json":
                # Convert to JSON
                result = df.to_json(orient="records", indent=2)
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(result)
                        
                return result
            
            # Calculate averages
            avg_peer_count = df["peer_count"].mean()
            avg_network_efficiency = df["network_efficiency"].mean()
            avg_network_density = df["network_density"].mean()
            avg_connections = df["average_connections"].mean()
            avg_optimization_score = df["optimization_score"].mean()
            
            # Count network health ratings
            health_counts = df["network_health"].value_counts()
            optimization_counts = df["optimization_rating"].value_counts()
            
            if format == "markdown":
                # Create markdown report
                report = "# P2P Network Metrics Report\n\n"
                report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                report += "## Summary\n\n"
                report += f"- Total tests: {len(df)}\n"
                report += f"- Average peer count: {avg_peer_count:.2f}\n"
                report += f"- Average network efficiency: {avg_network_efficiency:.2f}\n"
                report += f"- Average network density: {avg_network_density:.2f}\n"
                report += f"- Average connections per peer: {avg_connections:.2f}\n"
                report += f"- Average optimization score: {avg_optimization_score:.2f}\n\n"
                
                report += "## Network Health\n\n"
                for health, count in health_counts.items():
                    report += f"- {health}: {count} tests ({count/len(df)*100:.2f}%)\n"
                    
                report += "\n## Optimization Ratings\n\n"
                for rating, count in optimization_counts.items():
                    report += f"- {rating}: {count} tests ({count/len(df)*100:.2f}%)\n"
                
                report += "\n## Recent Tests\n\n"
                report += "| Model | Peers | Efficiency | Density | Optimization Score | Health | Timestamp |\n"
                report += "|-------|-------|------------|---------|-------------------|--------|----------|\n"
                
                for _, row in df.head(10).iterrows():
                    report += f"| {row['model_name']} | {row['peer_count']} | {row['network_efficiency']:.2f} | {row['network_density']:.2f} | {row['optimization_score']:.2f} | {row['network_health']} | {row['test_timestamp']} |\n"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            elif format == "html":
                # Create HTML report with interactive visualizations
                if not HAVE_PLOTLY:
                    return "Plotly not installed. Cannot generate HTML report."
                
                # Create network metrics radar chart
                categories = ['Peer Count', 'Network Efficiency', 'Network Density', 'Avg Connections', 'Optimization']
                
                # Normalize values for radar chart
                max_peer_count = df["peer_count"].max()
                normalized_peer_count = avg_peer_count / max_peer_count if max_peer_count > 0 else 0
                
                fig1 = go.Figure()
                
                fig1.add_trace(go.Scatterpolar(
                    r=[normalized_peer_count, avg_network_efficiency, avg_network_density, 
                       avg_connections / 5 if avg_connections > 0 else 0, avg_optimization_score],
                    theta=categories,
                    fill='toself',
                    name='Network Metrics'
                ))
                
                fig1.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="P2P Network Metrics Overview"
                )
                
                # Create optimization score vs efficiency scatter plot
                fig2 = px.scatter(
                    df,
                    x="network_efficiency",
                    y="optimization_score",
                    color="network_health",
                    size="peer_count",
                    hover_name="model_name",
                    title="Optimization Score vs Network Efficiency",
                    labels={"network_efficiency": "Network Efficiency", "optimization_score": "Optimization Score", "network_health": "Network Health"}
                )
                
                # Create health and optimization ratings pie charts
                fig3 = make_subplots(1, 2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                                     subplot_titles=["Network Health", "Optimization Ratings"])
                
                fig3.add_trace(
                    go.Pie(
                        labels=health_counts.index,
                        values=health_counts.values,
                        name="Network Health"
                    ),
                    row=1, col=1
                )
                
                fig3.add_trace(
                    go.Pie(
                        labels=optimization_counts.index,
                        values=optimization_counts.values,
                        name="Optimization Ratings"
                    ),
                    row=1, col=2
                )
                
                # Create HTML report
                report = "<html><head><title>P2P Network Metrics Report</title></head><body>"
                report += f"<h1>P2P Network Metrics Report</h1>"
                report += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
                
                report += "<h2>Summary</h2>"
                report += f"<p>Total tests: {len(df)}</p>"
                report += f"<p>Average peer count: {avg_peer_count:.2f}</p>"
                report += f"<p>Average network efficiency: {avg_network_efficiency:.2f}</p>"
                report += f"<p>Average network density: {avg_network_density:.2f}</p>"
                report += f"<p>Average connections per peer: {avg_connections:.2f}</p>"
                report += f"<p>Average optimization score: {avg_optimization_score:.2f}</p>"
                
                report += f"<div>{fig1.to_html(full_html=False)}</div>"
                report += f"<div>{fig2.to_html(full_html=False)}</div>"
                report += f"<div>{fig3.to_html(full_html=False)}</div>"
                
                report += "</body></html>"
                
                # Write to file if specified
                if output:
                    with open(output, "w") as f:
                        f.write(report)
                        
                return report
            
            else:
                return f"Unsupported format: {format}"
        
        except Exception as e:
            print(f"Error generating P2P report: {e}")
            traceback.print_exc()
            return f"Error generating P2P report: {e}"
