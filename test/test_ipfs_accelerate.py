import asyncio
import os
import sys
import json
import time
import traceback
from datetime import datetime
import importlib.util
from typing import Dict, List, Any, Optional, Union

# Set environment variables to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Determine if JSON output should be deprecated in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

# Set environment variable to avoid fork warnings in multiprocessing
# This helps prevent the "This process is multi-threaded, use of fork() may lead to deadlocks" warnings
# Reference: https://github.com/huggingface/transformers/issues/5486
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

# Configure to use spawn instead of fork to prevent deadlocks
import multiprocessing
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
    HAVE_DUCKDB = True
    print("DuckDB support enabled for test results")
except ImportError:
    HAVE_DUCKDB = False
    if DEPRECATE_JSON_OUTPUT:
        print("Warning: DuckDB not installed but DEPRECATE_JSON_OUTPUT=1. Will still save JSON as fallback.")
        print("To enable database storage, install duckdb: pip install duckdb pandas")


class TestResultsDBHandler:
    """
    Handler for storing test results in DuckDB database.
    This class abstracts away the database operations to store test results.
    Support for IPFS accelerator test results has been added.
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
            
            # Check if API is available
            self.api = None
            try:
                # Create a simple API wrapper for easier database queries
                # This helps with compatibility with other code that expects an API object
                class SimpleDBApi:
                    def __init__(self, conn):
                        self.conn = conn
                        
                    def query(self, query, params=None):
                        try:
                            if params:
                                result = self.conn.execute(query, params)
                            else:
                                result = self.conn.execute(query)
                            return result
                        except Exception as e:
                            print(f"Error executing query: {e}")
                            return None
                    
                    def execute_query(self, query, params=None):
                        return self.query(query, params)
                    
                    def store_integration_test_result(self, result):
                        # Simple implementation to store test results
                        try:
                            query = """
                            INSERT INTO test_runs (
                                run_id, test_name, test_type, success, started_at, 
                                completed_at, execution_time_seconds, metadata
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """
                            self.conn.execute(query, [
                                result.get("run_id", f"test_run_{int(time.time())}"),
                                result.get("test_name", result.get("test_module", "__test__")),
                                result.get("test_type", "integration"),
                                result.get("status", "pass") == "pass",
                                datetime.now(),
                                datetime.now(),
                                result.get("execution_time_seconds", 0),
                                json.dumps(result.get("metadata", {}))
                            ])
                            return True
                        except Exception as e:
                            print(f"Error storing integration test result: {e}")
                            return False
                    
                    def store_compatibility_result(self, result):
                        # Simple implementation to store compatibility results
                        try:
                            # Get model ID
                            model_id = None
                            model_query = "SELECT model_id FROM models WHERE model_name = ?"
                            model_result = self.conn.execute(model_query, [result.get("model_name")]).fetchone()
                            if model_result:
                                model_id = model_result[0]
                            else:
                                # Create model entry
                                self.conn.execute(
                                    "INSERT INTO models (model_name, model_family, added_at) VALUES (?, ?, ?)",
                                    [result.get("model_name"), result.get("model_family"), datetime.now()]
                                )
                                model_id = self.conn.execute(model_query, [result.get("model_name")]).fetchone()[0]
                            
                            # Get hardware ID
                            hardware_id = None
                            hardware_query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?"
                            hardware_result = self.conn.execute(hardware_query, [result.get("hardware_type")]).fetchone()
                            if hardware_result:
                                hardware_id = hardware_result[0]
                            else:
                                # Create hardware entry
                                self.conn.execute(
                                    "INSERT INTO hardware_platforms (hardware_type, device_name, detected_at) VALUES (?, ?, ?)",
                                    [result.get("hardware_type"), result.get("device_name", "unknown"), datetime.now()]
                                )
                                hardware_id = self.conn.execute(hardware_query, [result.get("hardware_type")]).fetchone()[0]
                            
                            # Store compatibility result
                            query = """
                            INSERT INTO hardware_compatibility (
                                model_id, hardware_id, compatibility_status, compatibility_score,
                                recommended, last_tested, run_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """
                            self.conn.execute(query, [
                                model_id,
                                hardware_id,
                                result.get("is_compatible", False),
                                result.get("score", 0.0),
                                result.get("recommended", False),
                                datetime.now(),
                                result.get("run_id", f"test_run_{int(time.time())}")
                            ])
                            return True
                        except Exception as e:
                            print(f"Error storing compatibility result: {e}")
                            return False
                
                # Create API instance
                self.api = SimpleDBApi(self.con)
                
            except Exception as e:
                print(f"Warning: Could not initialize database API layer: {e}")
                self.api = None
                
        except Exception as e:
            print(f"Warning: Failed to initialize database connection: {e}")
            self.con = None
            
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        if self.con is None:
            return
            
        try:
            # Create test_runs table for tracking test runs including IPFS accelerator tests
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    test_name VARCHAR,
                    test_type VARCHAR,
                    success BOOLEAN,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    execution_time_seconds FLOAT,
                    metadata VARCHAR
                )
            """)
            
            # Create ipfs_acceleration_results table for storing IPFS acceleration test results
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS ipfs_acceleration_results (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    model_name VARCHAR,
                    endpoint_type VARCHAR,
                    acceleration_type VARCHAR,
                    status VARCHAR,
                    success BOOLEAN,
                    execution_time_ms FLOAT,
                    implementation_type VARCHAR,
                    error_message VARCHAR,
                    additional_data VARCHAR,
                    test_date TIMESTAMP
                )
            """)
            
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
            
            # Create cross_platform_compatibility table
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS cross_platform_compatibility (
                    id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    model_family VARCHAR,
                    cuda_support BOOLEAN,
                    rocm_support BOOLEAN,
                    mps_support BOOLEAN,
                    openvino_support BOOLEAN,
                    qualcomm_support BOOLEAN,
                    webnn_support BOOLEAN,
                    webgpu_support BOOLEAN,
                    recommended_platform VARCHAR,
                    notes VARCHAR,
                    last_updated TIMESTAMP
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
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?", 
                [hardware_type, device_name]
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
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?", 
                [hardware_type, device_name]
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
                
            # Store model family information if available
            if 'model_family' in test_result and test_result['model_family']:
                self._update_model_family(model_id, test_result['model_family'])
                
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
                    test_id, model_id, hardware_id, 
                    power_watts_avg, power_watts_peak,
                    temperature_celsius_avg, temperature_celsius_peak,
                    battery_impact_mah, test_duration_seconds,
                    estimated_runtime_hours, test_timestamp
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
            
    def _update_model_family(self, model_id, model_family):
        """Update model family information in the database."""
        if self.con is None or not model_id or not model_family:
            return False
            
        try:
            # Update model family in the models table
            self.con.execute(
                """
                UPDATE models
                SET model_family = ?
                WHERE model_id = ?
                """,
                [model_family, model_id]
            )
            
            # Check if this family exists in the cross_platform_compatibility table
            family_query = """
            SELECT COUNT(*) 
            FROM cross_platform_compatibility
            WHERE model_family = ?
            """
            family_exists = self.con.execute(family_query, [model_family]).fetchone()[0] > 0
            
            # If it doesn't exist, create an entry
            if not family_exists:
                # Get model name from model_id
                model_name_query = """
                SELECT model_name
                FROM models
                WHERE model_id = ?
                """
                model_name = self.con.execute(model_name_query, [model_id]).fetchone()[0]
                
                # Create compatibility entry for the family
                now = datetime.now()
                self.con.execute(
                    """
                    INSERT INTO cross_platform_compatibility (
                        model_name, model_family, cuda_support, rocm_support,
                        mps_support, openvino_support, qualcomm_support,
                        webnn_support, webgpu_support, last_updated
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        model_name, model_family,
                        True, False, False, False, False, False, False,
                        now
                    ]
                )
            
            return True
        except Exception as e:
            print(f"Error updating model family: {e}")
            return False
            
    def generate_report(self, format='markdown', output_file=None):
        """Generate a report from the database."""
        if self.con is None:
            print("Cannot generate report - database connection not available")
            return None
            
        try:
            # Get summary data
            models_count = self.con.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            hardware_count = self.con.execute("SELECT COUNT(*) FROM hardware_platforms").fetchone()[0]
            tests_count = self.con.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]
            successful_tests = self.con.execute("SELECT COUNT(*) FROM test_results WHERE success = TRUE").fetchone()[0]
            
            # Get hardware platforms
            hardware_platforms = self.con.execute(
                "SELECT hardware_type, COUNT(*) FROM hardware_platforms GROUP BY hardware_type"
            ).fetchall()
            
            # Get model families
            model_families = self.con.execute(
                "SELECT model_family, COUNT(*) FROM models GROUP BY model_family"
            ).fetchall()
            
            # Get recent test results
            recent_tests = self.con.execute(
                """
                SELECT 
                    m.model_name, h.hardware_type, tr.status, tr.success, tr.timestamp
                FROM test_results tr
                JOIN models m ON tr.model_id = m.model_id
                JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                ORDER BY tr.timestamp DESC
                LIMIT 10
                """
            ).fetchall()
            
            # Get performance data
            performance_data = self.con.execute(
                """
                SELECT 
                    m.model_name, h.hardware_type, 
                    AVG(pr.average_latency_ms) as avg_latency,
                    AVG(pr.throughput_items_per_second) as avg_throughput,
                    AVG(pr.memory_peak_mb) as avg_memory
                FROM performance_results pr
                JOIN models m ON pr.model_id = m.model_id
                JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
                GROUP BY m.model_name, h.hardware_type
                ORDER BY m.model_name, avg_throughput DESC
                """
            ).fetchall()
            
            # Check if cross_platform_compatibility table exists and has data
            cross_platform_count = self.con.execute(
                "SELECT COUNT(*) FROM cross_platform_compatibility"
            ).fetchone()[0]
            
            if cross_platform_count > 0:
                # Use the dedicated cross-platform compatibility table
                compatibility_matrix = self.con.execute(
                    """
                    SELECT 
                        model_name,
                        model_family,
                        CASE WHEN cuda_support THEN 1 ELSE 0 END as cuda_support,
                        CASE WHEN rocm_support THEN 1 ELSE 0 END as rocm_support,
                        CASE WHEN mps_support THEN 1 ELSE 0 END as mps_support,
                        CASE WHEN openvino_support THEN 1 ELSE 0 END as openvino_support,
                        CASE WHEN qualcomm_support THEN 1 ELSE 0 END as qualcomm_support,
                        CASE WHEN webnn_support THEN 1 ELSE 0 END as webnn_support,
                        CASE WHEN webgpu_support THEN 1 ELSE 0 END as webgpu_support
                    FROM cross_platform_compatibility
                    ORDER BY model_family, model_name
                    """
                ).fetchall()
            else:
                # Fall back to generating matrix from test results
                compatibility_matrix = self.con.execute(
                    """
                    SELECT 
                        m.model_name,
                        m.model_family,
                        MAX(CASE WHEN h.hardware_type = 'cpu' THEN 1 ELSE 0 END) as cpu_support,
                        MAX(CASE WHEN h.hardware_type = 'cuda' THEN 1 ELSE 0 END) as cuda_support,
                        MAX(CASE WHEN h.hardware_type = 'rocm' THEN 1 ELSE 0 END) as rocm_support,
                        MAX(CASE WHEN h.hardware_type = 'mps' THEN 1 ELSE 0 END) as mps_support,
                        MAX(CASE WHEN h.hardware_type = 'openvino' THEN 1 ELSE 0 END) as openvino_support,
                        MAX(CASE WHEN h.hardware_type = 'qualcomm' THEN 1 ELSE 0 END) as qualcomm_support,
                        MAX(CASE WHEN h.hardware_type = 'webnn' THEN 1 ELSE 0 END) as webnn_support,
                        MAX(CASE WHEN h.hardware_type = 'webgpu' THEN 1 ELSE 0 END) as webgpu_support
                    FROM models m
                    LEFT JOIN test_results tr ON m.model_id = tr.model_id
                    LEFT JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                    GROUP BY m.model_name, m.model_family
                    """
                ).fetchall()
            
            # Format the report based on the requested format
            if format.lower() == 'markdown':
                report = self._generate_markdown_report(
                    models_count, hardware_count, tests_count, successful_tests,
                    hardware_platforms, model_families, recent_tests, 
                    performance_data, compatibility_matrix
                )
            elif format.lower() == 'html':
                report = self._generate_html_report(
                    models_count, hardware_count, tests_count, successful_tests,
                    hardware_platforms, model_families, recent_tests, 
                    performance_data, compatibility_matrix
                )
            elif format.lower() == 'json':
                report = self._generate_json_report(
                    models_count, hardware_count, tests_count, successful_tests,
                    hardware_platforms, model_families, recent_tests, 
                    performance_data, compatibility_matrix
                )
            else:
                print(f"Unsupported report format: {format}")
                return None
                
            # Write to file if output_file is specified
            if output_file and report:
                with open(output_file, 'w') as f:
                    f.write(report)
                print(f"Report written to {output_file}")
                
            return report
        except Exception as e:
            print(f"Error generating report: {e}")
            traceback.print_exc()
            return None
            
    def _generate_markdown_report(self, models_count, hardware_count, tests_count, successful_tests,
                                 hardware_platforms, model_families, recent_tests, 
                                 performance_data, compatibility_matrix):
        """Generate a markdown report from the database data."""
        report = []
        report.append("# IPFS Accelerate Test Results Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary section
        report.append("\n## Summary")
        report.append(f"- **Models**: {models_count}")
        report.append(f"- **Hardware Platforms**: {hardware_count}")
        report.append(f"- **Tests Run**: {tests_count}")
        success_rate = (successful_tests / tests_count * 100) if tests_count > 0 else 0
        report.append(f"- **Success Rate**: {success_rate:.2f}% ({successful_tests}/{tests_count})")
        
        # Hardware platforms section
        report.append("\n## Hardware Platforms")
        report.append("| Hardware Type | Count |")
        report.append("|--------------|-------|")
        for hw in hardware_platforms:
            report.append(f"| {hw[0] or 'Unknown'} | {hw[1]} |")
            
        # Model families section
        report.append("\n## Model Families")
        report.append("| Model Family | Count |")
        report.append("|-------------|-------|")
        for family in model_families:
            report.append(f"| {family[0] or 'Unknown'} | {family[1]} |")
            
        # Recent tests section
        report.append("\n## Recent Tests")
        report.append("| Model | Hardware | Status | Success | Timestamp |")
        report.append("|-------|----------|--------|---------|-----------|")
        for test in recent_tests:
            status_icon = "✅" if test[3] else "❌"
            report.append(f"| {test[0]} | {test[1]} | {test[2]} | {status_icon} | {test[4]} |")
            
        # Performance data section
        report.append("\n## Performance Data")
        report.append("| Model | Hardware | Avg Latency (ms) | Throughput (items/s) | Memory (MB) |")
        report.append("|-------|----------|------------------|---------------------|------------|")
        for perf in performance_data:
            report.append(f"| {perf[0]} | {perf[1]} | {perf[2]:.2f if perf[2] is not None else 'N/A'} | {perf[3]:.2f if perf[3] is not None else 'N/A'} | {perf[4]:.2f if perf[4] is not None else 'N/A'} |")
            
        # Compatibility matrix section
        report.append("\n## Hardware Compatibility Matrix")
        report.append("| Model | Family | CPU | CUDA | ROCm | MPS | OpenVINO | Qualcomm | WebNN | WebGPU |")
        report.append("|-------|--------|-----|------|------|-----|----------|----------|-------|--------|")
        for compat in compatibility_matrix:
            # Convert 1/0 to ✅/⚠️
            cpu = "✅" if compat[2] == 1 else "⚠️"
            cuda = "✅" if compat[3] == 1 else "⚠️"
            rocm = "✅" if compat[4] == 1 else "⚠️"
            mps = "✅" if compat[5] == 1 else "⚠️"
            openvino = "✅" if compat[6] == 1 else "⚠️"
            qualcomm = "✅" if compat[7] == 1 else "⚠️"
            webnn = "✅" if compat[8] == 1 else "⚠️"
            webgpu = "✅" if compat[9] == 1 else "⚠️"
            
            report.append(f"| {compat[0]} | {compat[1] or 'Unknown'} | {cpu} | {cuda} | {rocm} | {mps} | {openvino} | {qualcomm} | {webnn} | {webgpu} |")
            
        return "\n".join(report)
        
    def _generate_html_report(self, models_count, hardware_count, tests_count, successful_tests,
                             hardware_platforms, model_families, recent_tests, 
                             performance_data, compatibility_matrix):
        """Generate an HTML report from the database data."""
        # Basic HTML structure with some simple styling
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("    <title>IPFS Accelerate Test Results Report</title>")
        html.append("    <style>")
        html.append("        body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("        th { background-color: #f2f2f2; }")
        html.append("        tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append("        .success { color: green; }")
        html.append("        .failure { color: red; }")
        html.append("        .summary { display: flex; justify-content: space-between; flex-wrap: wrap; }")
        html.append("        .summary-box { border: 1px solid #ddd; padding: 15px; margin: 10px; min-width: 150px; text-align: center; }")
        html.append("        .summary-number { font-size: 24px; font-weight: bold; margin: 10px 0; }")
        html.append("    </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append(f"<h1>IPFS Accelerate Test Results Report</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary section with fancy boxes
        html.append("<h2>Summary</h2>")
        html.append("<div class='summary'>")
        html.append("    <div class='summary-box'>")
        html.append("        <div>Models</div>")
        html.append(f"        <div class='summary-number'>{models_count}</div>")
        html.append("    </div>")
        html.append("    <div class='summary-box'>")
        html.append("        <div>Hardware Platforms</div>")
        html.append(f"        <div class='summary-number'>{hardware_count}</div>")
        html.append("    </div>")
        html.append("    <div class='summary-box'>")
        html.append("        <div>Tests Run</div>")
        html.append(f"        <div class='summary-number'>{tests_count}</div>")
        html.append("    </div>")
        html.append("    <div class='summary-box'>")
        html.append("        <div>Success Rate</div>")
        success_rate = (successful_tests / tests_count * 100) if tests_count > 0 else 0
        html.append(f"        <div class='summary-number'>{success_rate:.2f}%</div>")
        html.append(f"        <div>({successful_tests}/{tests_count})</div>")
        html.append("    </div>")
        html.append("</div>")
        
        # Hardware platforms section
        html.append("<h2>Hardware Platforms</h2>")
        html.append("<table>")
        html.append("    <tr><th>Hardware Type</th><th>Count</th></tr>")
        for hw in hardware_platforms:
            html.append(f"    <tr><td>{hw[0] or 'Unknown'}</td><td>{hw[1]}</td></tr>")
        html.append("</table>")
        
        # Model families section
        html.append("<h2>Model Families</h2>")
        html.append("<table>")
        html.append("    <tr><th>Model Family</th><th>Count</th></tr>")
        for family in model_families:
            html.append(f"    <tr><td>{family[0] or 'Unknown'}</td><td>{family[1]}</td></tr>")
        html.append("</table>")
        
        # Recent tests section
        html.append("<h2>Recent Tests</h2>")
        html.append("<table>")
        html.append("    <tr><th>Model</th><th>Hardware</th><th>Status</th><th>Success</th><th>Timestamp</th></tr>")
        for test in recent_tests:
            success_class = "success" if test[3] else "failure"
            success_icon = "✅" if test[3] else "❌"
            html.append(f"    <tr><td>{test[0]}</td><td>{test[1]}</td><td>{test[2]}</td><td class='{success_class}'>{success_icon}</td><td>{test[4]}</td></tr>")
        html.append("</table>")
        
        # Performance data section
        html.append("<h2>Performance Data</h2>")
        html.append("<table>")
        html.append("    <tr><th>Model</th><th>Hardware</th><th>Avg Latency (ms)</th><th>Throughput (items/s)</th><th>Memory (MB)</th></tr>")
        for perf in performance_data:
            html.append(f"    <tr><td>{perf[0]}</td><td>{perf[1]}</td><td>{perf[2]:.2f if perf[2] is not None else 'N/A'}</td><td>{perf[3]:.2f if perf[3] is not None else 'N/A'}</td><td>{perf[4]:.2f if perf[4] is not None else 'N/A'}</td></tr>")
        html.append("</table>")
        
        # Compatibility matrix section
        html.append("<h2>Hardware Compatibility Matrix</h2>")
        html.append("<table>")
        html.append("    <tr><th>Model</th><th>Family</th><th>CPU</th><th>CUDA</th><th>ROCm</th><th>MPS</th><th>OpenVINO</th><th>Qualcomm</th><th>WebNN</th><th>WebGPU</th></tr>")
        for compat in compatibility_matrix:
            # Convert 1/0 to ✅/⚠️
            cpu = "✅" if compat[2] == 1 else "⚠️"
            cuda = "✅" if compat[3] == 1 else "⚠️"
            rocm = "✅" if compat[4] == 1 else "⚠️"
            mps = "✅" if compat[5] == 1 else "⚠️"
            openvino = "✅" if compat[6] == 1 else "⚠️"
            qualcomm = "✅" if compat[7] == 1 else "⚠️"
            webnn = "✅" if compat[8] == 1 else "⚠️"
            webgpu = "✅" if compat[9] == 1 else "⚠️"
            
            html.append(f"    <tr><td>{compat[0]}</td><td>{compat[1] or 'Unknown'}</td><td>{cpu}</td><td>{cuda}</td><td>{rocm}</td><td>{mps}</td><td>{openvino}</td><td>{qualcomm}</td><td>{webnn}</td><td>{webgpu}</td></tr>")
        html.append("</table>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
        
    def _generate_json_report(self, models_count, hardware_count, tests_count, successful_tests,
                             hardware_platforms, model_families, recent_tests, 
                             performance_data, compatibility_matrix):
        """Generate a JSON report from the database data."""
        # Convert tuples to lists for JSON serialization
        hardware_platforms_list = [{"hardware_type": hw[0], "count": hw[1]} for hw in hardware_platforms]
        model_families_list = [{"model_family": family[0], "count": family[1]} for family in model_families]
        
        recent_tests_list = [
            {
                "model": test[0],
                "hardware": test[1],
                "status": test[2],
                "success": bool(test[3]),
                "timestamp": str(test[4])
            }
            for test in recent_tests
        ]
        
        performance_data_list = [
            {
                "model": perf[0],
                "hardware": perf[1],
                "average_latency_ms": float(perf[2]) if perf[2] is not None else None,
                "throughput_items_per_second": float(perf[3]) if perf[3] is not None else None,
                "memory_peak_mb": float(perf[4]) if perf[4] is not None else None
            }
            for perf in performance_data
        ]
        
        compatibility_matrix_list = [
            {
                "model": compat[0],
                "family": compat[1],
                "cpu_support": bool(compat[2]),
                "cuda_support": bool(compat[3]),
                "rocm_support": bool(compat[4]),
                "mps_support": bool(compat[5]),
                "openvino_support": bool(compat[6]),
                "qualcomm_support": bool(compat[7]),
                "webnn_support": bool(compat[8]),
                "webgpu_support": bool(compat[9])
            }
            for compat in compatibility_matrix
        ]
        
        # Build the JSON structure
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "models_count": models_count,
                "hardware_count": hardware_count,
                "tests_count": tests_count,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / tests_count * 100) if tests_count > 0 else 0
            },
            "hardware_platforms": hardware_platforms_list,
            "model_families": model_families_list,
            "recent_tests": recent_tests_list,
            "performance_data": performance_data_list,
            "compatibility_matrix": compatibility_matrix_list
        }
        
        return json.dumps(report_data, indent=2)
    def generate_acceleration_comparison_report(self, format="html", output=None, model_name=None):
        """
        Generate a comparative report for acceleration types across different models or for a specific model.
        
        This report focuses on comparing the performance of different acceleration types (CUDA, OpenVINO, WebNN, etc.)
        to help users identify the best acceleration method for their use case.
        
        Args:
            format: Report format ("html", "json")
            output: Output file path (if None, returns the report as a string)
            model_name: Optional model name to filter results (if None, compares across all models)
            
        Returns:
            Report content as string if output is None, otherwise None
        """
        if not self.is_available():
            return "Database not available. Cannot generate comparison report."
        
        try:
            # Try to import visualization libraries
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd
                import numpy as np
                HAVE_PLOTLY = True
            except ImportError:
                HAVE_PLOTLY = False
                if format == "html":
                    print("Warning: plotly and pandas not available. Charts will not be generated.")
            
            # Query the ipfs_acceleration_results table for comparative data
            if model_name:
                # Query for a specific model
                query = """
                SELECT 
                    model_name, endpoint_type, acceleration_type, status, 
                    success, execution_time_ms, implementation_type,
                    test_date
                FROM 
                    ipfs_acceleration_results
                WHERE 
                    model_name = ?
                ORDER BY
                    test_date DESC
                """
                results = self.con.execute(query, [model_name]).fetchall()
                title = f"Acceleration Comparison for {model_name}"
            else:
                # Query across all models
                query = """
                SELECT 
                    model_name, endpoint_type, acceleration_type, status, 
                    success, execution_time_ms, implementation_type,
                    test_date
                FROM 
                    ipfs_acceleration_results
                ORDER BY
                    model_name, test_date DESC
                """
                results = self.con.execute(query).fetchall()
                title = "Acceleration Comparison Across All Models"
            
            if not results:
                return f"No acceleration results found{' for model '+model_name if model_name else ''}."
            
            # Process data for visualization
            acceleration_data = []
            for row in results:
                acceleration_data.append({
                    "Model": row[0],
                    "Endpoint Type": row[1],
                    "Acceleration Type": row[2] or "Unknown",
                    "Status": row[3] or "Unknown",
                    "Success": bool(row[4]),
                    "Execution Time (ms)": float(row[5]) if row[5] is not None else None,
                    "Implementation": row[6] or "Unknown",
                    "Test Date": row[7]
                })
            
            # Create DataFrame
            if not HAVE_PLOTLY:
                # Without visualization libraries, return text summary
                if format == "json":
                    return json.dumps({"acceleration_data": acceleration_data}, indent=2)
                else:
                    # Simple HTML table summary
                    html = ["<!DOCTYPE html><html><head><title>Acceleration Comparison</title>",
                           "<style>table {border-collapse: collapse; width: 100%;} th, td {padding: 8px; text-align: left; border: 1px solid #ddd;}</style>",
                           "</head><body>",
                           f"<h1>{title}</h1>",
                           "<table><tr><th>Model</th><th>Acceleration Type</th><th>Success</th><th>Execution Time (ms)</th></tr>"]
                    
                    for item in acceleration_data:
                        success_text = "✅" if item["Success"] else "❌"
                        time_text = f"{item['Execution Time (ms)']:.2f}" if item["Execution Time (ms)"] is not None else "N/A"
                        html.append(f"<tr><td>{item['Model']}</td><td>{item['Acceleration Type']}</td><td>{success_text}</td><td>{time_text}</td></tr>")
                    
                    html.append("</table></body></html>")
                    report = "\n".join(html)
                    
                    if output:
                        with open(output, "w") as f:
                            f.write(report)
                        return f"Report saved to {output}"
                    return report
            
            # With plotly, create rich visualizations
            df = pd.DataFrame(acceleration_data)
            
            # Prepare HTML report with visualizations
            html = []
            html.append("<!DOCTYPE html>")
            html.append("<html>")
            html.append("<head>")
            html.append(f"<title>{title}</title>")
            html.append("<style>")
            html.append("  body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; }")
            html.append("  .chart { width: 100%; height: 500px; margin-bottom: 30px; }")
            html.append("  .insight { background-color: #f8f9fa; border-left: 4px solid #4285f4; padding: 10px; margin: 15px 0; }")
            html.append("  h1, h2, h3 { color: #333; }")
            html.append("  table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
            html.append("  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html.append("  th { background-color: #f2f2f2; }")
            html.append("</style>")
            html.append("</head>")
            html.append("<body>")
            
            html.append(f"<h1>{title}</h1>")
            html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Create success rate comparison
            html.append("<h2>Success Rate by Acceleration Type</h2>")
            
            # Calculate success rates
            success_rates = df.groupby("Acceleration Type")["Success"].agg(
                ["count", "sum"]).reset_index()
            success_rates["Success Rate (%)"] = (success_rates["sum"] / 
                                              success_rates["count"] * 100).round(1)
            
            # Create success rate bar chart
            fig_success = px.bar(
                success_rates,
                x="Acceleration Type",
                y="Success Rate (%)",
                color="Success Rate (%)",
                title="Success Rate by Acceleration Type",
                color_continuous_scale=["#FF4136", "#FFDC00", "#2ECC40"],
                text="Success Rate (%)"
            )
            fig_success.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            
            html.append("<div class='chart'>")
            html.append(fig_success.to_html(full_html=False, include_plotlyjs='cdn'))
            html.append("</div>")
            
            # Add insights about success rates
            best_accel = success_rates.loc[success_rates["Success Rate (%)"].idxmax()]
            worst_accel = success_rates.loc[success_rates["Success Rate (%)"].idxmin()]
            
            html.append("<div class='insight'>")
            html.append("<h3>Success Rate Insights</h3>")
            html.append("<ul>")
            html.append(f"<li><strong>Most reliable acceleration:</strong> {best_accel['Acceleration Type']} " +
                       f"with {best_accel['Success Rate (%)']:.1f}% success rate ({best_accel['sum']}/{best_accel['count']} tests)</li>")
            html.append(f"<li><strong>Least reliable acceleration:</strong> {worst_accel['Acceleration Type']} " +
                       f"with {worst_accel['Success Rate (%)']:.1f}% success rate ({worst_accel['sum']}/{worst_accel['count']} tests)</li>")
            html.append("</ul>")
            html.append("</div>")
            
            # Performance comparison (only for successful tests)
            html.append("<h2>Performance Comparison</h2>")
            
            # Filter for successful tests with valid execution times
            df_success = df[(df["Success"] == True) & (df["Execution Time (ms)"].notna())]
            
            if not df_success.empty:
                # Create box plot of execution times
                fig_perf = px.box(
                    df_success,
                    x="Acceleration Type",
                    y="Execution Time (ms)",
                    color="Acceleration Type",
                    hover_data=["Model", "Implementation"],
                    title="Execution Time Distribution by Acceleration Type (Successful Tests Only)"
                )
                
                html.append("<div class='chart'>")
                html.append(fig_perf.to_html(full_html=False, include_plotlyjs='cdn'))
                html.append("</div>")
                
                # Add performance insights
                perf_stats = df_success.groupby("Acceleration Type")["Execution Time (ms)"].agg(
                    ["median", "mean", "std", "count"]).reset_index()
                
                fastest_median = perf_stats.loc[perf_stats["median"].idxmin()]
                slowest_median = perf_stats.loc[perf_stats["median"].idxmax()]
                
                html.append("<div class='insight'>")
                html.append("<h3>Performance Insights</h3>")
                html.append("<ul>")
                html.append(f"<li><strong>Fastest acceleration (median):</strong> {fastest_median['Acceleration Type']} " +
                           f"with {fastest_median['median']:.2f} ms median execution time</li>")
                html.append(f"<li><strong>Slowest acceleration (median):</strong> {slowest_median['Acceleration Type']} " +
                           f"with {slowest_median['median']:.2f} ms median execution time</li>")
                
                speed_diff = ((slowest_median['median'] - fastest_median['median']) / 
                             fastest_median['median'] * 100)
                html.append(f"<li><strong>Performance gap:</strong> {speed_diff:.1f}% slower</li>")
                html.append("</ul>")
                html.append("</div>")
                
                # If we have multiple models, add model-specific comparison
                if model_name is None and len(df_success["Model"].unique()) > 1:
                    # Create heatmap of median execution time by model and acceleration type
                    pivot_perf = df_success.pivot_table(
                        values="Execution Time (ms)",
                        index="Model",
                        columns="Acceleration Type",
                        aggfunc="median"
                    )
                    
                    fig_heatmap = px.imshow(
                        pivot_perf,
                        labels=dict(x="Acceleration Type", y="Model", color="Median Execution Time (ms)"),
                        title="Median Execution Time by Model and Acceleration Type (ms)",
                        color_continuous_scale=["#2ECC40", "#FFDC00", "#FF4136"]
                    )
                    
                    html.append("<div class='chart'>")
                    html.append(fig_heatmap.to_html(full_html=False, include_plotlyjs='cdn'))
                    html.append("</div>")
                    
                    # Find best acceleration type for each model
                    best_per_model = df_success.groupby(["Model", "Acceleration Type"])["Execution Time (ms)"].median().reset_index()
                    best_per_model = best_per_model.sort_values(["Model", "Execution Time (ms)"])
                    best_accel_by_model = best_per_model.groupby("Model").first().reset_index()
                    
                    # Create a summary table for best acceleration per model
                    html.append("<h3>Best Acceleration Type by Model</h3>")
                    html.append("<table>")
                    html.append("<tr><th>Model</th><th>Best Acceleration Type</th><th>Median Execution Time (ms)</th></tr>")
                    
                    for _, row in best_accel_by_model.iterrows():
                        html.append(f"<tr><td>{row['Model']}</td><td>{row['Acceleration Type']}</td><td>{row['Execution Time (ms)']:.2f}</td></tr>")
                    
                    html.append("</table>")
            else:
                html.append("<p>No successful tests with valid execution times found.</p>")
            
            # Add summary of available implementation types
            impl_counts = df.groupby(["Acceleration Type", "Implementation"]).size().reset_index(name="Count")
            
            if not impl_counts.empty:
                html.append("<h2>Implementation Types</h2>")
                html.append("<table>")
                html.append("<tr><th>Acceleration Type</th><th>Implementation</th><th>Count</th></tr>")
                
                for _, row in impl_counts.iterrows():
                    html.append(f"<tr><td>{row['Acceleration Type']}</td><td>{row['Implementation']}</td><td>{row['Count']}</td></tr>")
                
                html.append("</table>")
            
            # Close HTML
            html.append("</body>")
            html.append("</html>")
            
            # Assemble the report
            report = "\n".join(html)
            
            # Write to file if output is specified
            if output:
                with open(output, "w") as f:
                    f.write(report)
                return f"Report saved to {output}"
            
            return report
            
        except Exception as e:
            print(f"Error generating acceleration comparison report: {e}")
            traceback.print_exc()
            return f"Error generating comparison report: {str(e)}"
    
    def generate_ipfs_acceleration_report(self, format="markdown", output=None, run_id=None):
        """
        Generate a report specifically for IPFS acceleration results.
        
        Args:
            format: Report format ("markdown", "html", "json")
            output: Output file path (if None, returns the report as a string)
            run_id: Optional run ID to filter results (if None, uses latest run)
            
        Returns:
            Report content as string if output is None, otherwise None
        """
        if not self.is_available():
            return "Database not available. Cannot generate IPFS acceleration report."
            
        try:
            # Check if ipfs_acceleration_results table exists
            table_check = self.con.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='ipfs_acceleration_results'
            """).fetchone()
            
            if not table_check:
                # Create the table if it doesn't exist yet
                self.con.execute("""
                CREATE TABLE IF NOT EXISTS ipfs_acceleration_results (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    model_name VARCHAR,
                    endpoint_type VARCHAR,
                    acceleration_type VARCHAR,
                    status VARCHAR,
                    success BOOLEAN,
                    execution_time_ms FLOAT,
                    implementation_type VARCHAR,
                    error_message VARCHAR,
                    additional_data VARCHAR,
                    test_date TIMESTAMP
                )
                """)
                return "IPFS acceleration results table was created but contains no data yet."
            
            # Get run_id if not provided (use most recent)
            if run_id is None:
                run_query = "SELECT run_id FROM ipfs_acceleration_results ORDER BY test_date DESC LIMIT 1"
                run_result = self.con.execute(run_query).fetchone()
                if not run_result:
                    return "No IPFS acceleration test results found in database."
                run_id = run_result[0]
                
            # Get all acceleration results for this run
            query = """
            SELECT 
                model_name, endpoint_type, acceleration_type, status, 
                success, execution_time_ms, implementation_type,
                error_message, test_date
            FROM 
                ipfs_acceleration_results
            WHERE 
                run_id = ?
            ORDER BY
                model_name, endpoint_type
            """
            
            results = self.con.execute(query, [run_id]).fetchall()
            if not results:
                return f"No IPFS acceleration results found for run {run_id}"
                
            # Calculate summary statistics
            total_tests = len(results)
            successful_tests = sum(1 for r in results if r[4])  # r[4] is success boolean
            
            # Group by model
            model_results = {}
            for row in results:
                model = row[0]
                if model not in model_results:
                    model_results[model] = []
                model_results[model].append(row)
                
            # Group by acceleration type
            accel_results = {}
            for row in results:
                accel_type = row[2] or "Unknown"
                if accel_type not in accel_results:
                    accel_results[accel_type] = []
                accel_results[accel_type].append(row)
                
            # Calculate success rate by acceleration type
            accel_stats = {}
            for accel_type, rows in accel_results.items():
                total = len(rows)
                successful = sum(1 for r in rows if r[4])
                avg_time = 0
                if any(r[5] is not None for r in rows):
                    avg_time = sum(r[5] for r in rows if r[5] is not None) / sum(1 for r in rows if r[5] is not None)
                
                accel_stats[accel_type] = {
                    "total": total,
                    "successful": successful,
                    "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "N/A",
                    "avg_time_ms": avg_time if avg_time else "N/A"
                }
            
            # Generate report based on format
            if format.lower() == "markdown":
                return self._generate_ipfs_markdown_report(run_id, model_results, accel_stats, 
                                                         total_tests, successful_tests, output)
            elif format.lower() == "html":
                return self._generate_ipfs_html_report(run_id, model_results, accel_stats,
                                                     total_tests, successful_tests, output)
            elif format.lower() == "json":
                return self._generate_ipfs_json_report(run_id, model_results, accel_stats,
                                                     total_tests, successful_tests, output)
            else:
                return f"Unsupported format: {format}"
        
        except Exception as e:
            print(f"Error generating IPFS acceleration report: {e}")
            traceback.print_exc()
            return f"Error generating report: {str(e)}"
    
    def _generate_ipfs_markdown_report(self, run_id, model_results, accel_stats, total_tests, successful_tests, output=None):
        """Generate markdown report for IPFS acceleration results"""
        report = []
        
        # Header
        report.append("# IPFS Acceleration Test Results")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Run ID: {run_id}")
        report.append("")
        
        # Summary section
        report.append("## Summary")
        report.append(f"- **Total Tests**: {total_tests}")
        report.append(f"- **Successful Tests**: {successful_tests}")
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        report.append(f"- **Success Rate**: {success_rate:.1f}%")
        report.append("")
        
        # Performance by acceleration type
        report.append("## Performance by Acceleration Type")
        report.append("| Acceleration Type | Tests | Success Rate | Avg Execution Time (ms) |")
        report.append("|-------------------|-------|--------------|-------------------------|")
        
        for accel_type, stats in accel_stats.items():
            avg_time = stats["avg_time_ms"]
            avg_time_str = f"{avg_time:.2f}" if isinstance(avg_time, (int, float)) else avg_time
            report.append(f"| {accel_type} | {stats['total']} | {stats['success_rate']} | {avg_time_str} |")
        
        report.append("")
        
        # Results by model
        report.append("## Results by Model")
        
        for model, rows in model_results.items():
            report.append(f"### {model}")
            report.append("| Endpoint Type | Acceleration Type | Status | Success | Execution Time (ms) | Implementation |")
            report.append("|---------------|-------------------|--------|---------|---------------------|----------------|")
            
            for row in rows:
                endpoint = row[1]
                accel_type = row[2] or "Unknown"
                status = row[3] or "Unknown"
                success = "✅" if row[4] else "❌"
                time_ms = f"{row[5]:.2f}" if row[5] is not None else "N/A"
                impl_type = row[6] or "Unknown"
                
                report.append(f"| {endpoint} | {accel_type} | {status} | {success} | {time_ms} | {impl_type} |")
            
            report.append("")
        
        # Assemble the report
        report_text = "\n".join(report)
        
        # Write to file if output is specified
        if output:
            with open(output, "w") as f:
                f.write(report_text)
            return f"Report saved to {output}"
        
        return report_text
        
    def _generate_ipfs_html_report(self, run_id, model_results, accel_stats, total_tests, successful_tests, output=None):
        """Generate HTML report for IPFS acceleration results with enhanced visualizations"""
        html = []
        
        # Import necessary library
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            import numpy as np
            HAVE_PLOTLY = True
        except ImportError:
            HAVE_PLOTLY = False
        
        # HTML header and styles
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <title>IPFS Acceleration Test Results</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; }")
        html.append("    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("    th { background-color: #f2f2f2; }")
        html.append("    tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append("    .success { color: green; }")
        html.append("    .failure { color: red; }")
        html.append("    .summary { display: flex; justify-content: space-between; flex-wrap: wrap; }")
        html.append("    .summary-box { border: 1px solid #ddd; padding: 15px; margin: 10px; min-width: 150px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }")
        html.append("    .summary-number { font-size: 24px; font-weight: bold; margin: 10px 0; }")
        html.append("    h1, h2, h3 { color: #333; }")
        html.append("    .chart { width: 100%; height: 400px; margin-bottom: 30px; }")
        html.append("    .insights { background-color: #f8f9fa; border-left: 4px solid #4285f4; padding: 10px; margin: 15px 0; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Main header
        html.append("<h1>IPFS Acceleration Test Results</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p>Run ID: {run_id}</p>")
        
        # Summary section with fancy boxes
        html.append("<h2>Summary</h2>")
        html.append("<div class='summary'>")
        html.append("  <div class='summary-box'>")
        html.append("    <div>Total Tests</div>")
        html.append(f"    <div class='summary-number'>{total_tests}</div>")
        html.append("  </div>")
        html.append("  <div class='summary-box'>")
        html.append("    <div>Successful Tests</div>")
        html.append(f"    <div class='summary-number'>{successful_tests}</div>")
        html.append("  </div>")
        html.append("  <div class='summary-box'>")
        html.append("    <div>Success Rate</div>")
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        html.append(f"    <div class='summary-number'>{success_rate:.1f}%</div>")
        html.append("  </div>")
        html.append("</div>")
        
        # Add visualization for acceleration type statistics if plotly is available
        if HAVE_PLOTLY:
            # Create dataframe for acceleration stats
            accel_data = []
            for accel_type, stats in accel_stats.items():
                accel_data.append({
                    "Acceleration Type": accel_type,
                    "Total Tests": stats["total"],
                    "Success Rate": float(stats["success_rate"].replace("%", "")) if isinstance(stats["success_rate"], str) else stats["success_rate"],
                    "Avg Execution Time (ms)": stats["avg_time_ms"] if isinstance(stats["avg_time_ms"], (int, float)) else 0
                })
            
            if accel_data:
                df_accel = pd.DataFrame(accel_data)
                
                # Create bar chart for success rates
                fig_success = px.bar(
                    df_accel, 
                    x="Acceleration Type", 
                    y="Success Rate",
                    color="Success Rate",
                    labels={"Success Rate": "Success Rate (%)"},
                    title="Success Rate by Acceleration Type",
                    color_continuous_scale=["#FF4136", "#FFDC00", "#2ECC40"],
                    range_color=[0, 100]
                )
                
                # Create bar chart for execution times
                fig_time = px.bar(
                    df_accel, 
                    x="Acceleration Type", 
                    y="Avg Execution Time (ms)",
                    color="Avg Execution Time (ms)",
                    labels={"Avg Execution Time (ms)": "Avg Execution Time (ms)"},
                    title="Average Execution Time by Acceleration Type",
                    color_continuous_scale=["#2ECC40", "#FFDC00", "#FF4136"],
                )
                
                # Add the charts to the HTML report
                html.append("<div class='chart'>")
                html.append(fig_success.to_html(full_html=False, include_plotlyjs='cdn'))
                html.append("</div>")
                
                html.append("<div class='chart'>")
                html.append(fig_time.to_html(full_html=False, include_plotlyjs='cdn'))
                html.append("</div>")
                
                # Add insights section based on the data
                fastest_accel = df_accel.loc[df_accel["Avg Execution Time (ms)"].idxmin()]["Acceleration Type"] if len(df_accel) > 0 else "N/A"
                most_reliable = df_accel.loc[df_accel["Success Rate"].idxmax()]["Acceleration Type"] if len(df_accel) > 0 else "N/A"
                
                html.append("<div class='insights'>")
                html.append("<h3>Key Insights</h3>")
                html.append("<ul>")
                html.append(f"<li><strong>Fastest Acceleration Type:</strong> {fastest_accel}</li>")
                html.append(f"<li><strong>Most Reliable Acceleration Type:</strong> {most_reliable}</li>")
                if fastest_accel == most_reliable:
                    html.append(f"<li><strong>Recommendation:</strong> {fastest_accel} provides the best balance of speed and reliability.</li>")
                else:
                    html.append(f"<li><strong>Speed vs. Reliability Trade-off:</strong> {fastest_accel} is fastest but {most_reliable} is most reliable.</li>")
                html.append("</ul>")
                html.append("</div>")
        
        # Performance by acceleration type
        html.append("<h2>Performance by Acceleration Type</h2>")
        html.append("<table>")
        html.append("  <tr><th>Acceleration Type</th><th>Tests</th><th>Success Rate</th><th>Avg Execution Time (ms)</th></tr>")
        
        for accel_type, stats in accel_stats.items():
            avg_time = stats["avg_time_ms"]
            avg_time_str = f"{avg_time:.2f}" if isinstance(avg_time, (int, float)) else avg_time
            html.append(f"  <tr><td>{accel_type}</td><td>{stats['total']}</td><td>{stats['success_rate']}</td><td>{avg_time_str}</td></tr>")
        
        html.append("</table>")
        
        # Results by model
        html.append("<h2>Results by Model</h2>")
        
        # Process model results to create visualization data
        if HAVE_PLOTLY:
            model_data = []
            for model, rows in model_results.items():
                for row in rows:
                    endpoint = row[1]
                    accel_type = row[2] or "Unknown"
                    status = row[3] or "Unknown"
                    success = bool(row[4])
                    exec_time = float(row[5]) if row[5] is not None else None
                    impl_type = row[6] or "Unknown"
                    
                    model_data.append({
                        "Model": model,
                        "Endpoint Type": endpoint,
                        "Acceleration Type": accel_type,
                        "Status": status,
                        "Success": success,
                        "Execution Time (ms)": exec_time,
                        "Implementation Type": impl_type
                    })
            
            if model_data:
                df_models = pd.DataFrame(model_data)
                
                # Create heatmap of success rates by model and acceleration type
                success_pivot = pd.pivot_table(
                    df_models,
                    values="Success",
                    index="Model",
                    columns="Acceleration Type",
                    aggfunc=lambda x: 100 * sum(x) / len(x) if len(x) > 0 else 0
                )
                
                fig_heatmap = px.imshow(
                    success_pivot,
                    labels=dict(x="Acceleration Type", y="Model", color="Success Rate (%)"),
                    color_continuous_scale=["#FF4136", "#FFDC00", "#2ECC40"],
                    range_color=[0, 100],
                    title="Success Rate by Model and Acceleration Type (%)"
                )
                
                html.append("<div class='chart'>")
                html.append(fig_heatmap.to_html(full_html=False, include_plotlyjs='cdn'))
                html.append("</div>")
                
                # Create scatter plot of execution times by acceleration type
                # Only include successful tests with valid execution times
                df_success = df_models[(df_models["Success"] == True) & (df_models["Execution Time (ms)"].notna())]
                
                if not df_success.empty:
                    fig_scatter = px.box(
                        df_success,
                        x="Acceleration Type",
                        y="Execution Time (ms)",
                        color="Acceleration Type",
                        hover_data=["Model", "Implementation Type"],
                        title="Execution Time Distribution by Acceleration Type (Successful Tests Only)"
                    )
                    
                    html.append("<div class='chart'>")
                    html.append(fig_scatter.to_html(full_html=False, include_plotlyjs='cdn'))
                    html.append("</div>")
                    
                    # Add insights about execution time distribution
                    if len(df_success) > 0:
                        accel_median_times = df_success.groupby("Acceleration Type")["Execution Time (ms)"].median()
                        if not accel_median_times.empty:
                            fastest_accel = accel_median_times.idxmin()
                            slowest_accel = accel_median_times.idxmax()
                            time_diff_pct = ((accel_median_times[slowest_accel] - accel_median_times[fastest_accel]) / 
                                            accel_median_times[fastest_accel] * 100) if accel_median_times[fastest_accel] > 0 else 0
                            
                            html.append("<div class='insights'>")
                            html.append("<h3>Performance Insights</h3>")
                            html.append("<ul>")
                            html.append(f"<li><strong>Fastest median response time:</strong> {fastest_accel} ({accel_median_times[fastest_accel]:.2f} ms)</li>")
                            html.append(f"<li><strong>Slowest median response time:</strong> {slowest_accel} ({accel_median_times[slowest_accel]:.2f} ms)</li>")
                            html.append(f"<li><strong>Performance difference:</strong> {time_diff_pct:.1f}% slower</li>")
                            html.append("</ul>")
                            html.append("</div>")
        
        # Detailed results tables by model
        for model, rows in model_results.items():
            html.append(f"<h3>{model}</h3>")
            html.append("<table>")
            html.append("  <tr><th>Endpoint Type</th><th>Acceleration Type</th><th>Status</th><th>Success</th><th>Execution Time (ms)</th><th>Implementation</th></tr>")
            
            for row in rows:
                endpoint = row[1]
                accel_type = row[2] or "Unknown"
                status = row[3] or "Unknown"
                success_class = "success" if row[4] else "failure"
                success_icon = "✅" if row[4] else "❌"
                time_ms = f"{row[5]:.2f}" if row[5] is not None else "N/A"
                impl_type = row[6] or "Unknown"
                
                html.append(f"  <tr><td>{endpoint}</td><td>{accel_type}</td><td>{status}</td>" +
                          f"<td class='{success_class}'>{success_icon}</td><td>{time_ms}</td><td>{impl_type}</td></tr>")
            
            html.append("</table>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        # Assemble the report
        html_text = "\n".join(html)
        
        # Write to file if output is specified
        if output:
            with open(output, "w") as f:
                f.write(html_text)
            return f"Report saved to {output}"
        
        return html_text
        
    def _generate_ipfs_json_report(self, run_id, model_results, accel_stats, total_tests, successful_tests, output=None):
        """Generate JSON report for IPFS acceleration results"""
        # Convert model results to JSON-friendly format
        model_results_json = {}
        for model, rows in model_results.items():
            model_results_json[model] = [
                {
                    "endpoint_type": row[1],
                    "acceleration_type": row[2] or "Unknown",
                    "status": row[3] or "Unknown",
                    "success": bool(row[4]),
                    "execution_time_ms": row[5],
                    "implementation_type": row[6] or "Unknown",
                    "error_message": row[7],
                    "test_date": str(row[8])
                }
                for row in rows
            ]
        
        # Create report data structure
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "run_id": run_id,
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "acceleration_stats": accel_stats,
            "model_results": model_results_json
        }
        
        # Convert to JSON
        json_text = json.dumps(report_data, indent=2)
        
        # Write to file if output is specified
        if output:
            with open(output, "w") as f:
                f.write(json_text)
            return f"Report saved to {output}"
        
        return json_text
    
    def is_available(self) -> bool:
        """Check if database storage is available."""
        return HAVE_DUCKDB and self.con is not None
    
    def store_test_results(self, results: Dict[str, Any], run_id: str = None) -> bool:
        """
        Store test results in the database.
        
        Args:
            results: Test results dictionary
            run_id: Optional run ID to associate results with
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_available():
            return False
            
        try:
            # Generate run_id if not provided
            if run_id is None:
                run_id = f"test_run_{int(time.time())}"
                
            # Store integration test results
            self._store_integration_results(results, run_id)
            
            # Store hardware compatibility results
            self._store_compatibility_results(results, run_id)
            
            # Store power metrics if available (mainly for Qualcomm devices)
            self._store_power_metrics(results, run_id)
            
            # Store IPFS acceleration results if present
            self._store_ipfs_acceleration_results(results, run_id)
            
            return True
        except Exception as e:
            print(f"Error storing test results in database: {e}")
            print(traceback.format_exc())
            return False
            
    def _store_ipfs_acceleration_results(self, results: Dict[str, Any], run_id: str):
        """Extract and store IPFS acceleration results from the test results dictionary."""
        if not self.is_available() or not results:
            return
            
        # Look for IPFS acceleration test results in different places in the results structure
        if "ipfs_accelerate_tests" in results and isinstance(results["ipfs_accelerate_tests"], dict):
            # Process model results to extract IPFS acceleration data
            for model, model_data in results["ipfs_accelerate_tests"].items():
                # Skip summary entry
                if model == "summary":
                    continue
                    
                # Check for local endpoints (CUDA, OpenVINO)
                if "local_endpoint" in model_data and isinstance(model_data["local_endpoint"], dict):
                    for endpoint_type, endpoint_results in model_data["local_endpoint"].items():
                        # Only process endpoints if they exist and are dictionaries
                        if isinstance(endpoint_results, dict):
                            # Store the endpoint results in the IPFS acceleration table
                            self.store_ipfs_acceleration_result(
                                model_name=model,
                                endpoint_type=endpoint_type,
                                acceleration_results=endpoint_results,
                                run_id=run_id
                            )
                
                # Check for qualcomm endpoint
                if "qualcomm_endpoint" in model_data and isinstance(model_data["qualcomm_endpoint"], dict):
                    # Store the Qualcomm endpoint results
                    self.store_ipfs_acceleration_result(
                        model_name=model,
                        endpoint_type="qualcomm",
                        acceleration_results=model_data["qualcomm_endpoint"],
                        run_id=run_id
                    )
                    
                # Check for API endpoints
                if "api_endpoint" in model_data and isinstance(model_data["api_endpoint"], dict):
                    for endpoint_type, endpoint_results in model_data["api_endpoint"].items():
                        # Process only if results are a dictionary
                        if isinstance(endpoint_results, dict):
                            # Store the API endpoint results
                            self.store_ipfs_acceleration_result(
                                model_name=model,
                                endpoint_type=f"api_{endpoint_type}",
                                acceleration_results=endpoint_results,
                                run_id=run_id
                            )
                            
                # Check for WebNN endpoints
                if "webnn_endpoint" in model_data and isinstance(model_data["webnn_endpoint"], dict):
                    # Store the WebNN endpoint results
                    self.store_ipfs_acceleration_result(
                        model_name=model,
                        endpoint_type="webnn",
                        acceleration_results=model_data["webnn_endpoint"],
                        run_id=run_id
                    )
                    
        # Also look for direct test results structure
        elif "test_endpoints" in results and isinstance(results["test_endpoints"], dict):
            for model, endpoint_data in results["test_endpoints"].items():
                # Skip non-model entries
                if model in ["test_stats", "endpoint_handler_resources"]:
                    continue
                    
                # Process different endpoint types
                for endpoint_key in ["local_endpoint", "qualcomm_endpoint", "api_endpoint", "webnn_endpoint"]:
                    if endpoint_key in endpoint_data and isinstance(endpoint_data[endpoint_key], dict):
                        # For local endpoints, we may have multiple endpoint types
                        if endpoint_key == "local_endpoint":
                            for endpoint_type, endpoint_results in endpoint_data[endpoint_key].items():
                                if isinstance(endpoint_results, dict):
                                    self.store_ipfs_acceleration_result(
                                        model_name=model,
                                        endpoint_type=endpoint_type,
                                        acceleration_results=endpoint_results,
                                        run_id=run_id
                                    )
                        else:
                            # Direct endpoint type
                            endpoint_type = endpoint_key.replace("_endpoint", "")
                            self.store_ipfs_acceleration_result(
                                model_name=model,
                                endpoint_type=endpoint_type,
                                acceleration_results=endpoint_data[endpoint_key],
                                run_id=run_id
                            )
            
    def execute_query(self, query: str, params: list = None):
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            
        Returns:
            Query result or False if error occurred
        """
        if not self.is_available():
            return False
            
        try:
            if params:
                result = self.con.execute(query, params)
            else:
                result = self.con.execute(query)
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return False
            
    def get_test_results(self, run_id=None, model=None, hardware_type=None, limit=50):
        """
        Query test results from the database with flexible filtering.
        
        Args:
            run_id: Optional run ID to filter results
            model: Optional model name to filter results
            hardware_type: Optional hardware type to filter results
            limit: Maximum number of results to return (default 50)
            
        Returns:
            Query result or None if query failed
        """
        if not self.is_available():
            return None
            
        try:
            # Base query with appropriate joins
            query = """
            SELECT 
                tr.id, tr.timestamp, tr.test_date, tr.status, tr.test_type,
                m.model_name, m.model_family, 
                hp.hardware_type, hp.device_name,
                tr.success, tr.error_message, tr.execution_time,
                tr.memory_usage, pr.batch_size, pr.average_latency_ms, 
                pr.throughput_items_per_second, pr.memory_peak_mb
            FROM 
                test_results tr
            LEFT JOIN 
                models m ON tr.model_id = m.model_id
            LEFT JOIN 
                hardware_platforms hp ON tr.hardware_id = hp.hardware_id
            LEFT JOIN 
                performance_results pr ON tr.id = pr.id
            """
            
            # Build WHERE clause
            where_clauses = []
            params = []
            
            if run_id:
                where_clauses.append("tr.id = ?")
                params.append(run_id)
                
            if model:
                where_clauses.append("m.model_name = ?")
                params.append(model)
                
            if hardware_type:
                where_clauses.append("hp.hardware_type = ?")
                params.append(hardware_type)
                
            # Add WHERE clause if we have any conditions
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            # Add ORDER BY and LIMIT
            query += " ORDER BY tr.timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            return self.execute_query(query, params)
            
        except Exception as e:
            print(f"Error getting test results: {e}")
            return None
            
    def generate_report(self, format="markdown", output_file=None):
        """
        Generate a comprehensive report from database results.
        
        Args:
            format: Report format ("markdown", "html", "json")
            output_file: Output file path (if None, returns the report as a string)
            
        Returns:
            Report content as string if output_file is None, otherwise None
        """
        if not self.is_available():
            return "Database not available. Cannot generate report."
            
        try:
            # Get summary data
            models_count = self.con.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            hardware_count = self.con.execute("SELECT COUNT(*) FROM hardware_platforms").fetchone()[0]
            tests_count = self.con.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]
            successful_tests = self.con.execute("SELECT COUNT(*) FROM test_results WHERE success = TRUE").fetchone()[0]
            
            # Get hardware platforms
            hardware_platforms = self.con.execute(
                "SELECT hardware_type, COUNT(*) FROM hardware_platforms GROUP BY hardware_type"
            ).fetchall()
            
            # Get model families
            model_families = self.con.execute(
                "SELECT model_family, COUNT(*) FROM models GROUP BY model_family"
            ).fetchall()
            
            # Get recent test results
            recent_tests = self.con.execute(
                """
                SELECT 
                    m.model_name, h.hardware_type, tr.status, tr.success, tr.timestamp
                FROM test_results tr
                JOIN models m ON tr.model_id = m.model_id
                JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                ORDER BY tr.timestamp DESC
                LIMIT 10
                """
            ).fetchall()
            
            # Get performance data
            performance_data = self.con.execute(
                """
                SELECT 
                    m.model_name, h.hardware_type, 
                    AVG(pr.average_latency_ms) as avg_latency,
                    AVG(pr.throughput_items_per_second) as avg_throughput,
                    AVG(pr.memory_peak_mb) as avg_memory
                FROM performance_results pr
                JOIN models m ON pr.model_id = m.model_id
                JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
                GROUP BY m.model_name, h.hardware_type
                ORDER BY m.model_name, avg_throughput DESC
                """
            ).fetchall()
            
            # Check if cross_platform_compatibility table exists and has data
            cross_platform_count = self.con.execute(
                "SELECT COUNT(*) FROM cross_platform_compatibility"
            ).fetchone()[0]
            
            if cross_platform_count > 0:
                # Use the dedicated cross-platform compatibility table
                compatibility_matrix = self.con.execute(
                    """
                    SELECT 
                        model_name,
                        model_family,
                        CASE WHEN cuda_support THEN 1 ELSE 0 END as cuda_support,
                        CASE WHEN rocm_support THEN 1 ELSE 0 END as rocm_support,
                        CASE WHEN mps_support THEN 1 ELSE 0 END as mps_support,
                        CASE WHEN openvino_support THEN 1 ELSE 0 END as openvino_support,
                        CASE WHEN qualcomm_support THEN 1 ELSE 0 END as qualcomm_support,
                        CASE WHEN webnn_support THEN 1 ELSE 0 END as webnn_support,
                        CASE WHEN webgpu_support THEN 1 ELSE 0 END as webgpu_support
                    FROM cross_platform_compatibility
                    ORDER BY model_family, model_name
                    """
                ).fetchall()
            else:
                # Fall back to generating matrix from test results
                compatibility_matrix = self.con.execute(
                    """
                    SELECT 
                        m.model_name,
                        m.model_family,
                        MAX(CASE WHEN h.hardware_type = 'cpu' THEN 1 ELSE 0 END) as cpu_support,
                        MAX(CASE WHEN h.hardware_type = 'cuda' THEN 1 ELSE 0 END) as cuda_support,
                        MAX(CASE WHEN h.hardware_type = 'rocm' THEN 1 ELSE 0 END) as rocm_support,
                        MAX(CASE WHEN h.hardware_type = 'mps' THEN 1 ELSE 0 END) as mps_support,
                        MAX(CASE WHEN h.hardware_type = 'openvino' THEN 1 ELSE 0 END) as openvino_support,
                        MAX(CASE WHEN h.hardware_type = 'qualcomm' THEN 1 ELSE 0 END) as qualcomm_support,
                        MAX(CASE WHEN h.hardware_type = 'webnn' THEN 1 ELSE 0 END) as webnn_support,
                        MAX(CASE WHEN h.hardware_type = 'webgpu' THEN 1 ELSE 0 END) as webgpu_support
                    FROM models m
                    LEFT JOIN test_results tr ON m.model_id = tr.model_id
                    LEFT JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                    GROUP BY m.model_name, m.model_family
                    """
                ).fetchall()
            
            # Format the report based on the requested format
            if format.lower() == 'markdown':
                report = self._generate_markdown_report(
                    models_count, hardware_count, tests_count, successful_tests,
                    hardware_platforms, model_families, recent_tests, 
                    performance_data, compatibility_matrix
                )
            elif format.lower() == 'html':
                report = self._generate_html_report(
                    models_count, hardware_count, tests_count, successful_tests,
                    hardware_platforms, model_families, recent_tests, 
                    performance_data, compatibility_matrix
                )
            elif format.lower() == 'json':
                report = self._generate_json_report(
                    models_count, hardware_count, tests_count, successful_tests,
                    hardware_platforms, model_families, recent_tests, 
                    performance_data, compatibility_matrix
                )
            else:
                return f"Unsupported format: {format}"
            
            # Write to file if output_file is specified
            if output_file and report:
                with open(output_file, 'w') as f:
                    f.write(report)
                return f"Report saved to {output_file}"
            else:
                return report
                
        except Exception as e:
            print(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating report: {e}"
            
    def _generate_markdown_report(self, models_count, hardware_count, tests_count, successful_tests,
                                 hardware_platforms, model_families, recent_tests, 
                                 performance_data, compatibility_matrix):
        """Generate a markdown report from the database data."""
        report = []
        report.append("# IPFS Accelerate Test Results Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary section
        report.append("\n## Summary")
        report.append(f"- **Models**: {models_count}")
        report.append(f"- **Hardware Platforms**: {hardware_count}")
        report.append(f"- **Tests Run**: {tests_count}")
        success_rate = (successful_tests / tests_count * 100) if tests_count > 0 else 0
        report.append(f"- **Success Rate**: {success_rate:.2f}% ({successful_tests}/{tests_count})")
        
        # Hardware platforms section
        report.append("\n## Hardware Platforms")
        report.append("| Hardware Type | Count |")
        report.append("|--------------|-------|")
        for hw in hardware_platforms:
            report.append(f"| {hw[0] or 'Unknown'} | {hw[1]} |")
            
        # Model families section
        report.append("\n## Model Families")
        report.append("| Model Family | Count |")
        report.append("|-------------|-------|")
        for family in model_families:
            report.append(f"| {family[0] or 'Unknown'} | {family[1]} |")
            
        # Recent tests section
        report.append("\n## Recent Tests")
        report.append("| Model | Hardware | Status | Success | Timestamp |")
        report.append("|-------|----------|--------|---------|-----------|")
        for test in recent_tests:
            status_icon = "✅" if test[3] else "❌"
            report.append(f"| {test[0]} | {test[1]} | {test[2]} | {status_icon} | {test[4]} |")
            
        # Performance data section
        report.append("\n## Performance Data")
        report.append("| Model | Hardware | Avg Latency (ms) | Throughput (items/s) | Memory (MB) |")
        report.append("|-------|----------|------------------|---------------------|------------|")
        for perf in performance_data:
            report.append(f"| {perf[0]} | {perf[1]} | {"N/A" if perf[2] is None else f"{perf[2]:.2f}"} | {"N/A" if perf[3] is None else f"{perf[3]:.2f}"} | {"N/A" if perf[4] is None else f"{perf[4]:.2f}"} |")
            
        # Compatibility matrix section
        report.append("\n## Hardware Compatibility Matrix")
        report.append("| Model | Family | CPU | CUDA | ROCm | MPS | OpenVINO | Qualcomm | WebNN | WebGPU |")
        report.append("|-------|--------|-----|------|------|-----|----------|----------|-------|--------|")
        for compat in compatibility_matrix:
            # Convert 1/0 to ✅/⚠️
            cpu = "✅" if compat[2] == 1 else "⚠️"
            cuda = "✅" if compat[3] == 1 else "⚠️"
            rocm = "✅" if compat[4] == 1 else "⚠️"
            mps = "✅" if compat[5] == 1 else "⚠️"
            openvino = "✅" if compat[6] == 1 else "⚠️"
            qualcomm = "✅" if compat[7] == 1 else "⚠️"
            webnn = "✅" if compat[8] == 1 else "⚠️"
            webgpu = "✅" if compat[9] == 1 else "⚠️"
            
            report.append(f"| {compat[0]} | {compat[1] or 'Unknown'} | {cpu} | {cuda} | {rocm} | {mps} | {openvino} | {qualcomm} | {webnn} | {webgpu} |")
            
        return "\n".join(report)
    
    def _generate_html_report(self, models_count, hardware_count, tests_count, successful_tests,
                             hardware_platforms, model_families, recent_tests, 
                             performance_data, compatibility_matrix):
        """Generate an HTML report from the database data."""
        # Import necessary library
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            HAVE_PLOTLY = True
        except ImportError:
            HAVE_PLOTLY = False
        
        # Basic HTML structure
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("  <title>IPFS Accelerate Python Test Report</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }")
        html.append("    h1, h2 { color: #333; }")
        html.append("    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("    th { background-color: #f2f2f2; }")
        html.append("    tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append("    .chart { width: 100%; height: 400px; margin-bottom: 30px; }")
        html.append("    .success { color: green; }")
        html.append("    .failure { color: red; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append("<h1>IPFS Accelerate Python Test Report</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Run information
        html.append("<h2>Test Run Information</h2>")
        html.append("<table>")
        html.append("  <tr><th>Property</th><th>Value</th></tr>")
        html.append(f"  <tr><td>Test Name</td><td>{run_info.get('test_name', 'Database Report')}</td></tr>")
        html.append(f"  <tr><td>Test Type</td><td>{run_info.get('test_type', 'Database')}</td></tr>")
        html.append(f"  <tr><td>Started</td><td>{run_info.get('started_at', 'Unknown')}</td></tr>")
        html.append(f"  <tr><td>Completed</td><td>{run_info.get('completed_at', 'Unknown')}</td></tr>")
        html.append(f"  <tr><td>Execution Time</td><td>{run_info.get('execution_time_seconds', 0):.2f} seconds</td></tr>")
        success_class = "success" if run_info.get('success', False) else "failure"
        html.append(f"  <tr><td>Success</td><td class='{success_class}'>{run_info.get('success', False)}</td></tr>")
        html.append("</table>")
        
        # Hardware compatibility
        html.append("<h2>Hardware Compatibility</h2>")
        html.append("<table>")
        html.append("  <tr><th>Model</th><th>Hardware Type</th><th>Device Name</th><th>Compatible</th>" +
                   "<th>Detection</th><th>Initialization</th><th>Error</th></tr>")
        
        for item in compatibility:
            compatible_class = "success" if item.get('is_compatible', False) else "failure"
            detection_class = "success" if item.get('detection_success', False) else "failure"
            init_class = "success" if item.get('initialization_success', False) else "failure"
            
            html.append(f"  <tr>")
            html.append(f"    <td>{item.get('model_name', 'Unknown')}</td>")
            html.append(f"    <td>{item.get('hardware_type', 'Unknown')}</td>")
            html.append(f"    <td>{item.get('device_name', 'Unknown')}</td>")
            html.append(f"    <td class='{compatible_class}'>{item.get('is_compatible', False)}</td>")
            html.append(f"    <td class='{detection_class}'>{item.get('detection_success', False)}</td>")
            html.append(f"    <td class='{init_class}'>{item.get('initialization_success', False)}</td>")
            html.append(f"    <td>{item.get('error_message', 'None')}</td>")
            html.append(f"  </tr>")
        html.append("</table>")
        
        # Add compatibility chart if plotly is available
        if HAVE_PLOTLY:
            df = pd.DataFrame(compatibility)
            if not df.empty:
                # Create a pivot table for compatibility by model and hardware
                pivot_df = pd.crosstab(
                    index=df['model_name'], 
                    columns=df['hardware_type'], 
                    values=df['is_compatible'],
                    aggfunc=lambda x: 1 if x.any() else 0
                )
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df, 
                    labels=dict(x="Hardware Type", y="Model", color="Compatible"),
                    x=pivot_df.columns, 
                    y=pivot_df.index,
                    color_continuous_scale=["#FF4136", "#2ECC40"],
                    range_color=[0, 1]
                )
                fig.update_layout(title="Model-Hardware Compatibility Matrix")
                
                # Add the chart to the HTML
                html.append("<div class='chart'>")
                html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                html.append("</div>")
        
        # Performance metrics
        if performance:
            html.append("<h2>Performance Metrics</h2>")
            html.append("<table>")
            html.append("  <tr><th>Model</th><th>Hardware Type</th><th>Batch Size</th><th>Precision</th>" +
                       "<th>Latency (ms)</th><th>Throughput (items/s)</th><th>Memory (MB)</th></tr>")
            
            for item in performance:
                html.append(f"  <tr>")
                html.append(f"    <td>{item.get('model_name', 'Unknown')}</td>")
                html.append(f"    <td>{item.get('hardware_type', 'Unknown')}</td>")
                html.append(f"    <td>{item.get('batch_size', 1)}</td>")
                html.append(f"    <td>{item.get('precision', 'Unknown')}</td>")
                html.append(f"    <td>{item.get('average_latency_ms', 0):.2f}</td>")
                html.append(f"    <td>{item.get('throughput_items_per_second', 0):.2f}</td>")
                html.append(f"    <td>{item.get('memory_peak_mb', 0):.2f}</td>")
                html.append(f"  </tr>")
            html.append("</table>")
            
            # Add performance chart if plotly is available
            if HAVE_PLOTLY:
                df = pd.DataFrame(performance)
                if not df.empty:
                    # Bar chart for throughput comparison
                    fig = px.bar(
                        df, 
                        x='model_name', 
                        y='throughput_items_per_second', 
                        color='hardware_type',
                        barmode='group',
                        labels={'model_name': 'Model', 'throughput_items_per_second': 'Throughput (items/s)', 'hardware_type': 'Hardware'},
                        title='Throughput Comparison by Model and Hardware'
                    )
                    
                    # Add the chart to the HTML
                    html.append("<div class='chart'>")
                    html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                    html.append("</div>")
        
        # Power metrics
        if power_metrics:
            html.append("<h2>Power Metrics</h2>")
            html.append("<table>")
            html.append("  <tr><th>Model</th><th>Hardware Type</th><th>Power (mW)</th><th>Energy (mJ)</th>" +
                       "<th>Temperature (°C)</th><th>Efficiency (items/J)</th><th>Battery Impact (%/h)</th></tr>")
            
            for item in power_metrics:
                html.append(f"  <tr>")
                html.append(f"    <td>{item.get('model_name', 'Unknown')}</td>")
                html.append(f"    <td>{item.get('hardware_type', 'Unknown')}</td>")
                html.append(f"    <td>{item.get('power_consumption_mw', 0):.2f}</td>")
                html.append(f"    <td>{item.get('energy_consumption_mj', 0):.2f}</td>")
                html.append(f"    <td>{item.get('temperature_celsius', 0):.2f}</td>")
                html.append(f"    <td>{item.get('energy_efficiency_items_per_joule', 0):.2f}</td>")
                html.append(f"    <td>{item.get('battery_impact_percent_per_hour', 0):.2f}</td>")
                html.append(f"  </tr>")
            html.append("</table>")
            
            # Add efficiency chart if plotly is available
            if HAVE_PLOTLY:
                df = pd.DataFrame(power_metrics)
                if not df.empty:
                    # Bar chart for energy efficiency
                    fig = px.bar(
                        df, 
                        x='model_name', 
                        y='energy_efficiency_items_per_joule', 
                        color='hardware_type',
                        barmode='group',
                        labels={'model_name': 'Model', 'energy_efficiency_items_per_joule': 'Efficiency (items/J)', 'hardware_type': 'Hardware'},
                        title='Energy Efficiency Comparison'
                    )
                    
                    # Add the chart to the HTML
                    html.append("<div class='chart'>")
                    html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                    html.append("</div>")
        
        # Summary
        html.append("<h2>Summary</h2>")
        
        # Calculate compatibility rate
        compatible_count = sum(1 for item in compatibility if item.get('is_compatible', False))
        compatibility_rate = (compatible_count / len(compatibility) * 100) if compatibility else 0
        
        # Calculate metrics for summary
        unique_models = len(set(item.get('model_name') for item in compatibility))
        unique_hardware = len(set(item.get('hardware_type') for item in compatibility))
        
        html.append("<ul>")
        html.append(f"<li><strong>Models Tested:</strong> {unique_models}</li>")
        html.append(f"<li><strong>Hardware Platforms:</strong> {unique_hardware}</li>")
        html.append(f"<li><strong>Compatibility Rate:</strong> {compatibility_rate:.1f}%</li>")
        
        if performance:
            # Find best performing hardware
            best_hardware = {}
            for item in performance:
                model = item.get('model_name')
                hardware = item.get('hardware_type')
                throughput = item.get('throughput_items_per_second', 0)
                
                if model not in best_hardware or throughput > best_hardware[model]['throughput']:
                    best_hardware[model] = {
                        'hardware': hardware,
                        'throughput': throughput
                    }
            
            if best_hardware:
                html.append("<li><strong>Best Performing Hardware by Model:</strong>")
                html.append("<ul>")
                for model, info in best_hardware.items():
                    html.append(f"<li>{model}: {info['hardware']} ({info['throughput']:.2f} items/s)</li>")
                html.append("</ul>")
                html.append("</li>")
        
        html.append("</ul>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
            
    def store_ipfs_acceleration_result(self, model_name, endpoint_type, acceleration_results, run_id=None):
        """
        Store IPFS acceleration test results in the database.
        
        Args:
            model_name: The name of the model being tested
            endpoint_type: The endpoint type (cuda, openvino, etc.)
            acceleration_results: Results from the acceleration test
            run_id: Optional run ID to associate results with
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_available():
            return False
            
        try:
            # Create the table if it doesn't exist
            self.con.execute("""
            CREATE TABLE IF NOT EXISTS ipfs_acceleration_results (
                id INTEGER PRIMARY KEY,
                run_id VARCHAR,
                model_name VARCHAR,
                endpoint_type VARCHAR,
                acceleration_type VARCHAR,
                status VARCHAR,
                success BOOLEAN,
                execution_time_ms FLOAT,
                implementation_type VARCHAR,
                error_message VARCHAR,
                additional_data VARCHAR,
                test_date TIMESTAMP
            )
            """)
            
            # Generate run_id if not provided
            if run_id is None:
                run_id = f"ipfs_accel_{int(time.time())}"
                
            now = datetime.now()
            
            # Determine if the test was successful
            success = False
            status = "Unknown"
            error_message = None
            execution_time = None
            implementation_type = "Unknown"
            additional_data = {}
            
            # Extract data based on result structure
            if isinstance(acceleration_results, dict):
                # Get status directly or infer from other fields
                if "status" in acceleration_results:
                    status = acceleration_results["status"]
                    success = status.lower() == "success"
                    
                # Get error message if present
                if "error" in acceleration_results:
                    error_message = acceleration_results["error"]
                elif "error_message" in acceleration_results:
                    error_message = acceleration_results["error_message"]
                
                # Get execution time if present
                if "execution_time_ms" in acceleration_results:
                    execution_time = acceleration_results["execution_time_ms"]
                elif "execution_time" in acceleration_results:
                    execution_time = acceleration_results["execution_time"]
                    
                # Get implementation type if present
                if "implementation_type" in acceleration_results:
                    implementation_type = acceleration_results["implementation_type"]
                    
                # Store any additional data as JSON
                additional_data = {k: v for k, v in acceleration_results.items() 
                                 if k not in ["status", "error", "error_message", 
                                             "execution_time_ms", "execution_time",
                                             "implementation_type"]}
            
            # Determine acceleration type based on endpoint_type
            acceleration_type = "Unknown"
            if "cuda" in endpoint_type.lower():
                acceleration_type = "GPU"
            elif "openvino" in endpoint_type.lower():
                acceleration_type = "CPU"
            elif "webgpu" in endpoint_type.lower():
                acceleration_type = "WebGPU"
            elif "webnn" in endpoint_type.lower():
                acceleration_type = "WebNN"
            elif "qualcomm" in endpoint_type.lower():
                acceleration_type = "Mobile"
                
            # Insert the result into the database
            self.con.execute("""
                INSERT INTO ipfs_acceleration_results (
                    run_id, model_name, endpoint_type, acceleration_type, status,
                    success, execution_time_ms, implementation_type, error_message,
                    additional_data, test_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, model_name, endpoint_type, acceleration_type, status,
                success, execution_time, implementation_type, error_message,
                json.dumps(additional_data), now
            ])
            
            # Print success message
            print(f"Stored IPFS acceleration result for {model_name} on {endpoint_type} (acceleration type: {acceleration_type})")
            
            return True
        except Exception as e:
            print(f"Error storing IPFS acceleration result: {e}")
            traceback.print_exc()
            return False
    
    def _store_power_metrics(self, results: Dict[str, Any], run_id: str):
        """Store power and thermal metrics in the dedicated power_metrics table."""
        # Skip if no results or API not available or execute_query is not available
        if not self.is_available() or not results or not hasattr(self.api, "execute_query"):
            return
            
        # Check if ipfs_accelerate_tests exists and has model results
        if "ipfs_accelerate_tests" not in results or not isinstance(results["ipfs_accelerate_tests"], dict):
            return
            
        model_results = results["ipfs_accelerate_tests"]
        for model, model_data in model_results.items():
            # Skip summary entry
            if model == "summary":
                continue
                
            # Look for power metrics in both types of endpoints
            endpoints = {}
            
            # Process local endpoints for power metrics
            if "local_endpoint" in model_data and isinstance(model_data["local_endpoint"], dict):
                endpoints.update(model_data["local_endpoint"])
                
            # Process qualcomm_endpoint if it exists
            if "qualcomm_endpoint" in model_data and isinstance(model_data["qualcomm_endpoint"], dict):
                endpoints["qualcomm"] = model_data["qualcomm_endpoint"]
            
            # Process each endpoint that might have power metrics
            for endpoint_type, endpoint_data in endpoints.items():
                # Skip if not a valid endpoint type or data is not a dict
                if not isinstance(endpoint_data, dict):
                    continue
                    
                # Look for power metrics in different places
                power_metrics = {}
                if "power_metrics" in endpoint_data and isinstance(endpoint_data["power_metrics"], dict):
                    power_metrics = endpoint_data["power_metrics"]
                elif "metrics" in endpoint_data and isinstance(endpoint_data["metrics"], dict):
                    metrics = endpoint_data["metrics"]
                    
                    # Standard power metric fields
                    standard_fields = [
                        "power_consumption_mw", "energy_consumption_mj", "temperature_celsius", 
                        "monitoring_duration_ms", "average_power_mw", "peak_power_mw", "idle_power_mw"
                    ]
                    
                    # Enhanced metric fields
                    enhanced_fields = [
                        "energy_efficiency_items_per_joule", "thermal_throttling_detected",
                        "battery_impact_percent_per_hour", "model_type"
                    ]
                    
                    # Extract all available fields
                    for key in standard_fields + enhanced_fields:
                        if key in metrics:
                            power_metrics[key] = metrics[key]
                
                # Skip if no power metrics found
                if not power_metrics:
                    continue
                
                # Determine hardware type
                hardware_type = "cpu"  # Default
                if "cuda" in endpoint_type.lower():
                    hardware_type = "cuda"
                elif "openvino" in endpoint_type.lower():
                    hardware_type = "openvino" 
                elif "qualcomm" in endpoint_type.lower() or endpoint_type == "qualcomm":
                    hardware_type = "qualcomm"
                    
                # Get device info if available
                device_name = None
                sdk_type = None
                sdk_version = None
                
                if "device_info" in endpoint_data and isinstance(endpoint_data["device_info"], dict):
                    device_info = endpoint_data["device_info"]
                    device_name = device_info.get("device_name")
                    sdk_type = device_info.get("sdk_type")
                    sdk_version = device_info.get("sdk_version")
                
                # Extract model type from different possible locations
                model_type = power_metrics.get("model_type")
                if not model_type and "model_type" in endpoint_data:
                    model_type = endpoint_data["model_type"]
                if not model_type and "device_info" in endpoint_data and "model_type" in endpoint_data["device_info"]:
                    model_type = endpoint_data["device_info"]["model_type"]
                
                # Get throughput info if available
                throughput = None
                throughput_units = None
                if "throughput" in endpoint_data:
                    throughput = endpoint_data["throughput"]
                if "throughput_units" in endpoint_data:
                    throughput_units = endpoint_data["throughput_units"]
                
                # Handle the special case of thermal_throttling_detected being a boolean
                thermal_throttling = power_metrics.get("thermal_throttling_detected")
                if isinstance(thermal_throttling, str):
                    thermal_throttling = thermal_throttling.lower() in ["true", "yes", "1"]
                
                # Prepare SQL parameters for enhanced schema
                params = [
                    run_id,
                    model,
                    hardware_type,
                    power_metrics.get("power_consumption_mw"),
                    power_metrics.get("energy_consumption_mj"),
                    power_metrics.get("temperature_celsius"),
                    power_metrics.get("monitoring_duration_ms"),
                    power_metrics.get("average_power_mw"),
                    power_metrics.get("peak_power_mw"),
                    power_metrics.get("idle_power_mw"),
                    device_name,
                    sdk_type,
                    sdk_version,
                    model_type,
                    power_metrics.get("energy_efficiency_items_per_joule"),
                    thermal_throttling,
                    power_metrics.get("battery_impact_percent_per_hour"),
                    throughput,
                    throughput_units,
                    json.dumps(power_metrics)
                ]
                
                # Create the SQL query with enhanced fields
                query = """
                INSERT INTO power_metrics (
                    run_id, model_name, hardware_type, 
                    power_consumption_mw, energy_consumption_mj, temperature_celsius,
                    monitoring_duration_ms, average_power_mw, peak_power_mw, idle_power_mw,
                    device_name, sdk_type, sdk_version, model_type,
                    energy_efficiency_items_per_joule, thermal_throttling_detected,
                    battery_impact_percent_per_hour, throughput, throughput_units,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Execute the query
                try:
                    self.api.execute_query(query, params)
                    print(f"Stored enhanced power metrics for {model} ({model_type}) on {hardware_type}")
                except Exception as e:
                    print(f"Error storing power metrics for {model} on {hardware_type}: {e}")
    
    def _store_integration_results(self, results: Dict[str, Any], run_id: str):
        """Store integration test results in the database."""
        # Skip if no results or API not available
        if not self.is_available() or not results:
            return
            
        # Create integration test result for main test
        main_result = {
            "test_module": "test_ipfs_accelerate",
            "test_class": "test_ipfs_accelerate",
            "test_name": "__test__",
            "status": results.get("status", "unknown"),
            "execution_time_seconds": results.get("execution_time", 0),
            "run_id": run_id,
            "metadata": {
                "timestamp": results.get("timestamp", ""),
                "test_date": results.get("test_date", "")
            }
        }
        
        try:
            self.api.store_integration_test_result(main_result)
        except Exception as e:
            print(f"Error storing main integration test result: {e}")
            
        # Store results for each model tested
        if "ipfs_accelerate_tests" in results and isinstance(results["ipfs_accelerate_tests"], dict):
            model_results = results["ipfs_accelerate_tests"]
            
            for model, model_data in model_results.items():
                # Skip summary entry
                if model == "summary":
                    continue
                    
                # Create test result for this model
                model_status = "pass" if model_data.get("status") == "Success" else "fail"
                model_result = {
                    "test_module": "test_ipfs_accelerate",
                    "test_class": "test_ipfs_accelerate.model_test",
                    "test_name": f"test_{model}",
                    "status": model_status,
                    "model_name": model,
                    "run_id": run_id,
                    "metadata": model_data
                }
                
                # Store model test result
                try:
                    self.api.store_integration_test_result(model_result)
                except Exception as e:
                    print(f"Error storing model test result for {model}: {e}")
    
    def _store_compatibility_results(self, results: Dict[str, Any], run_id: str):
        """Store hardware compatibility results in the database."""
        # Skip if no results or API not available
        if not self.is_available() or not results:
            return
            
        # Check if ipfs_accelerate_tests exists and has model results
        if "ipfs_accelerate_tests" not in results or not isinstance(results["ipfs_accelerate_tests"], dict):
            return
            
        model_results = results["ipfs_accelerate_tests"]
        for model, model_data in model_results.items():
            # Skip summary entry
            if model == "summary":
                continue
                
            # Process hardware compatibility results for local endpoints (CUDA, OpenVINO)
            if "local_endpoint" in model_data and isinstance(model_data["local_endpoint"], dict):
                for endpoint_type, endpoint_data in model_data["local_endpoint"].items():
                    # Skip if not a valid endpoint type or data is not a dict
                    if not isinstance(endpoint_data, dict):
                        continue
                        
                    # Determine hardware type
                    hardware_type = "cpu"  # Default
                    if "cuda" in endpoint_type.lower():
                        hardware_type = "cuda"
                    elif "openvino" in endpoint_type.lower():
                        hardware_type = "openvino"
                    elif "qualcomm" in endpoint_type.lower():
                        hardware_type = "qualcomm"
                    
                    # Extract power and thermal metrics if available (mainly for Qualcomm)
                    power_metrics = {}
                    if "power_metrics" in endpoint_data and isinstance(endpoint_data["power_metrics"], dict):
                        power_metrics = endpoint_data["power_metrics"]
                    elif "metrics" in endpoint_data and isinstance(endpoint_data["metrics"], dict):
                        # Try to extract from metrics field too
                        metrics = endpoint_data["metrics"]
                        for key in ["power_consumption_mw", "energy_consumption_mj", "temperature_celsius"]:
                            if key in metrics:
                                power_metrics[key] = metrics[key]
                    
                    # Create compatibility result with power metrics
                    compatibility = {
                        "model_name": model,
                        "hardware_type": hardware_type,
                        "is_compatible": endpoint_data.get("status", "").lower() == "success",
                        "detection_success": True,
                        "initialization_success": not ("error" in endpoint_data or "error_message" in endpoint_data),
                        "error_message": endpoint_data.get("error", endpoint_data.get("error_message", "")),
                        "run_id": run_id,
                        "metadata": {
                            "implementation_type": endpoint_data.get("implementation_type", "unknown"),
                            "endpoint_type": endpoint_type,
                            "power_consumption_mw": power_metrics.get("power_consumption_mw"),
                            "energy_consumption_mj": power_metrics.get("energy_consumption_mj"),
                            "temperature_celsius": power_metrics.get("temperature_celsius"),
                            "monitoring_duration_ms": power_metrics.get("monitoring_duration_ms")
                        }
                    }
                    
                    # Store compatibility result
                    try:
                        self.api.store_compatibility_result(compatibility)
                    except Exception as e:
                        print(f"Error storing compatibility result for {model} on {hardware_type}: {e}")
                        
    def generate_webgpu_analysis_report(self, format='markdown', output=None, browser=None, 
                                       include_shader_metrics=False, analyze_compute_shaders=False):
        """Generate a WebGPU performance analysis report from the database."""
        if self.con is None:
            print("Cannot generate WebGPU analysis report - database connection not available")
            return None
            
        try:
            # Get WebGPU test data
            webgpu_data = self.con.execute("""
                SELECT 
                    m.model_name, 
                    tr.test_date, 
                    tr.success, 
                    tr.execution_time,
                    tr.details
                FROM test_results tr
                JOIN models m ON tr.model_id = m.model_id
                JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                WHERE h.hardware_type = 'webgpu'
                ORDER BY tr.timestamp DESC
            """).fetchall()
            
            # Get browser-specific WebGPU performance if browser is specified
            browser_specific_data = None
            if browser:
                browser_specific_data = self.con.execute("""
                    SELECT 
                        m.model_name, 
                        tr.test_date, 
                        tr.success, 
                        tr.execution_time,
                        tr.details
                    FROM test_results tr
                    JOIN models m ON tr.model_id = m.model_id
                    JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                    WHERE h.hardware_type = 'webgpu' AND tr.details LIKE ?
                    ORDER BY tr.timestamp DESC
                """, [f'%"browser": "{browser}"%']).fetchall()
            
            # Get shader metrics if requested
            shader_metrics = None
            if include_shader_metrics:
                try:
                    shader_metrics = self.con.execute("""
                        SELECT 
                            m.model_name,
                            json_extract(tr.details, '$.shader_compilation_time_ms') as compilation_time,
                            json_extract(tr.details, '$.shader_count') as shader_count,
                            json_extract(tr.details, '$.shader_cache_hits') as cache_hits,
                            tr.test_date
                        FROM test_results tr
                        JOIN models m ON tr.model_id = m.model_id
                        JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                        WHERE h.hardware_type = 'webgpu' 
                          AND tr.details LIKE '%shader_compilation_time_ms%'
                        ORDER BY tr.timestamp DESC
                    """).fetchall()
                except Exception as e:
                    print(f"Warning: Could not extract shader metrics: {e}")
                    shader_metrics = []
            
            # Get compute shader data if requested
            compute_shader_data = None
            if analyze_compute_shaders:
                try:
                    compute_shader_data = self.con.execute("""
                        SELECT 
                            m.model_name,
                            json_extract(tr.details, '$.compute_shader_optimization') as optimization,
                            json_extract(tr.details, '$.execution_time_ms') as execution_time,
                            json_extract(tr.details, '$.browser') as browser,
                            tr.test_date
                        FROM test_results tr
                        JOIN models m ON tr.model_id = m.model_id
                        JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
                        WHERE h.hardware_type = 'webgpu' 
                          AND tr.details LIKE '%compute_shader_optimization%'
                        ORDER BY tr.timestamp DESC
                    """).fetchall()
                except Exception as e:
                    print(f"Warning: Could not extract compute shader data: {e}")
                    compute_shader_data = []
            
            # Get WebGPU vs other hardware comparison
            comparison_data = self.con.execute("""
                SELECT 
                    m.model_name,
                    h.hardware_type,
                    AVG(pr.average_latency_ms) as avg_latency,
                    AVG(pr.throughput_items_per_second) as avg_throughput,
                    AVG(pr.memory_peak_mb) as avg_memory
                FROM performance_results pr
                JOIN models m ON pr.model_id = m.model_id
                JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
                GROUP BY m.model_name, h.hardware_type
                ORDER BY m.model_name, avg_throughput DESC
            """).fetchall()
            
            # Format the report based on the requested format
            if format.lower() == 'markdown':
                report = self._generate_webgpu_markdown_report(
                    webgpu_data, browser_specific_data, shader_metrics, 
                    compute_shader_data, comparison_data, browser
                )
            elif format.lower() == 'html':
                report = self._generate_webgpu_html_report(
                    webgpu_data, browser_specific_data, shader_metrics, 
                    compute_shader_data, comparison_data, browser
                )
            elif format.lower() == 'json':
                report = self._generate_webgpu_json_report(
                    webgpu_data, browser_specific_data, shader_metrics, 
                    compute_shader_data, comparison_data, browser
                )
            else:
                print(f"Unsupported report format: {format}")
                return None
                
            # Write to file if output is specified
            if output and report:
                with open(output, 'w') as f:
                    f.write(report)
                print(f"WebGPU analysis report written to {output}")
                
            return report
        except Exception as e:
            print(f"Error generating WebGPU analysis report: {e}")
            traceback.print_exc()
            return None
            
    def _generate_webgpu_markdown_report(self, webgpu_data, browser_specific_data, 
                                       shader_metrics, compute_shader_data, comparison_data, browser=None):
        """Generate a markdown WebGPU analysis report."""
        report = []
        report.append("# WebGPU Performance Analysis Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if browser:
            report.append(f"\nBrowser: {browser}")
        
        # WebGPU overall performance section
        report.append("\n## WebGPU Performance Overview")
        if not webgpu_data:
            report.append("\nNo WebGPU test data found in the database.")
        else:
            report.append(f"\nFound {len(webgpu_data)} WebGPU test results.")
            report.append("\n| Model | Test Date | Success | Execution Time (ms) |")
            report.append("|-------|-----------|---------|---------------------|")
            for data in webgpu_data[:10]:  # Show top 10 results
                status_icon = "✅" if data[2] else "❌"
                report.append(f"| {data[0]} | {data[1]} | {status_icon} | {data[3] if data[3] is not None else 'N/A'} |")
            
            # Success rate calculation
            success_count = sum(1 for data in webgpu_data if data[2])
            success_rate = (success_count / len(webgpu_data) * 100) if webgpu_data else 0
            report.append(f"\n**Success Rate**: {success_rate:.2f}% ({success_count}/{len(webgpu_data)})")
        
        # Browser-specific section
        if browser_specific_data:
            report.append(f"\n## {browser} WebGPU Performance")
            report.append(f"\nFound {len(browser_specific_data)} test results for {browser}.")
            report.append("\n| Model | Test Date | Success | Execution Time (ms) |")
            report.append("|-------|-----------|---------|---------------------|")
            for data in browser_specific_data[:10]:  # Show top 10 results
                status_icon = "✅" if data[2] else "❌"
                report.append(f"| {data[0]} | {data[1]} | {status_icon} | {data[3] if data[3] is not None else 'N/A'} |")
        
        # Shader metrics section
        if shader_metrics:
            report.append("\n## Shader Compilation Metrics")
            report.append("\n| Model | Compilation Time (ms) | Shader Count | Cache Hits | Test Date |")
            report.append("|-------|------------------------|--------------|------------|-----------|")
            for metric in shader_metrics[:10]:  # Show top 10 results
                report.append(f"| {metric[0]} | {metric[1] if metric[1] is not None else 'N/A'} | " +
                             f"{metric[2] if metric[2] is not None else 'N/A'} | " +
                             f"{metric[3] if metric[3] is not None else 'N/A'} | {metric[4]} |")
        
        # Compute shader optimization section
        if compute_shader_data:
            report.append("\n## Compute Shader Optimization Analysis")
            report.append("\n| Model | Optimization | Execution Time (ms) | Browser | Test Date |")
            report.append("|-------|--------------|---------------------|---------|-----------|")
            for data in compute_shader_data[:10]:  # Show top 10 results
                report.append(f"| {data[0]} | {data[1] if data[1] is not None else 'N/A'} | " +
                             f"{data[2] if data[2] is not None else 'N/A'} | " +
                             f"{data[3] if data[3] is not None else 'N/A'} | {data[4]} |")
        
        # WebGPU vs other hardware comparison
        if comparison_data:
            report.append("\n## WebGPU vs Other Hardware")
            report.append("\n| Model | Hardware | Avg Latency (ms) | Throughput (items/s) | Memory (MB) |")
            report.append("|-------|----------|------------------|---------------------|------------|")
            
            # Group by model
            model_data = {}
            for data in comparison_data:
                model = data[0]
                if model not in model_data:
                    model_data[model] = []
                model_data[model].append(data)
            
            # Output comparison for each model
            for model, data_points in model_data.items():
                for data in data_points:
                    report.append(f"| {data[0]} | {data[1]} | {data[2]:.2f if data[2] is not None else 'N/A'} | " +
                                 f"{data[3]:.2f if data[3] is not None else 'N/A'} | " +
                                 f"{data[4]:.2f if data[4] is not None else 'N/A'} |")
        
        # Recommendations section
        report.append("\n## Recommendations")
        
        # Add general recommendations
        report.append("\n### General Recommendations")
        report.append("- Use shader precompilation to improve initial load times")
        report.append("- Consider WebGPU for models that are compatible with browser environment")
        report.append("- Test with multiple browsers to optimize for specific user environments")
        
        # Add browser-specific recommendations if available
        if browser:
            report.append(f"\n### {browser}-Specific Recommendations")
            if browser.lower() == "chrome":
                report.append("- Ensure latest Chrome version is used for best WebGPU performance")
                report.append("- For compute-heavy workloads, consider balanced workgroup sizes (e.g., 128x2x1)")
            elif browser.lower() == "firefox":
                report.append("- For audio models, Firefox often shows 20-30% better performance with compute shaders")
                report.append("- Use 256x1x1 workgroup size for best performance with audio models")
            elif browser.lower() == "safari":
                report.append("- WebGPU support in Safari has limitations; test thoroughly")
                report.append("- Fall back to WebNN for broader compatibility on Safari")
            elif browser.lower() == "edge":
                report.append("- Edge performs similarly to Chrome for most WebGPU workloads")
                report.append("- Consider enabling hardware acceleration in browser settings")
        
        return "\n".join(report)

    def _generate_webgpu_html_report(self, webgpu_data, browser_specific_data, 
                                   shader_metrics, compute_shader_data, comparison_data, browser=None):
        """Generate an HTML WebGPU analysis report."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("    <title>WebGPU Performance Analysis Report</title>")
        html.append("    <style>")
        html.append("        body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("        th { background-color: #f2f2f2; }")
        html.append("        tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append("        .success { color: green; }")
        html.append("        .failure { color: red; }")
        html.append("        .summary { display: flex; justify-content: space-between; flex-wrap: wrap; }")
        html.append("        .summary-box { border: 1px solid #ddd; padding: 15px; margin: 10px; min-width: 150px; text-align: center; }")
        html.append("        .summary-number { font-size: 24px; font-weight: bold; margin: 10px 0; }")
        html.append("        .recommendation { background-color: #f8f9fa; padding: 10px; border-left: 4px solid #4285f4; margin: 15px 0; }")
        html.append("    </style>")
        html.append("    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>")
        html.append("</head>")
        html.append("<body>")
        html.append("    <h1>WebGPU Performance Analysis Report</h1>")
        html.append(f"    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        if browser:
            html.append(f"    <p>Browser: {browser}</p>")
        
        # WebGPU overall performance section
        html.append("    <h2>WebGPU Performance Overview</h2>")
        if not webgpu_data:
            html.append("    <p>No WebGPU test data found in the database.</p>")
        else:
            # Calculate success metrics
            success_count = sum(1 for data in webgpu_data if data[2])
            success_rate = (success_count / len(webgpu_data) * 100) if webgpu_data else 0
            
            # Add summary boxes
            html.append("    <div class=\"summary\">")
            html.append("        <div class=\"summary-box\">")
            html.append("            <h3>Total Tests</h3>")
            html.append(f"            <div class=\"summary-number\">{len(webgpu_data)}</div>")
            html.append("        </div>")
            html.append("        <div class=\"summary-box\">")
            html.append("            <h3>Success Rate</h3>")
            html.append(f"            <div class=\"summary-number\">{success_rate:.1f}%</div>")
            html.append("        </div>")
            html.append("        <div class=\"summary-box\">")
            html.append("            <h3>Successful Tests</h3>")
            html.append(f"            <div class=\"summary-number\">{success_count}</div>")
            html.append("        </div>")
            html.append("    </div>")
            
            # Add test results table
            html.append("    <h3>Recent WebGPU Tests</h3>")
            html.append("    <table>")
            html.append("        <tr><th>Model</th><th>Test Date</th><th>Success</th><th>Execution Time (ms)</th></tr>")
            for data in webgpu_data[:10]:  # Show top 10 results
                status_icon = "<span class=\"success\">✅</span>" if data[2] else "<span class=\"failure\">❌</span>"
                html.append(f"        <tr><td>{data[0]}</td><td>{data[1]}</td><td>{status_icon}</td><td>{data[3] if data[3] is not None else 'N/A'}</td></tr>")
            html.append("    </table>")
            
            # Add visualization div for performance chart
            html.append("    <div id=\"performance-chart\" style=\"width:100%; height:400px;\"></div>")
            
            # Add JavaScript for chart
            html.append("    <script>")
            html.append("        // Prepare data for chart")
            html.append("        const performanceData = {")
            html.append("            models: " + json.dumps([data[0] for data in webgpu_data[:10] if data[2] and data[3] is not None]) + ",")
            html.append("            times: " + json.dumps([data[3] for data in webgpu_data[:10] if data[2] and data[3] is not None]) + "")
            html.append("        };")
            html.append("        ")
            html.append("        // Create chart")
            html.append("        if (performanceData.models.length > 0) {")
            html.append("            const trace = {")
            html.append("                x: performanceData.models,")
            html.append("                y: performanceData.times,")
            html.append("                type: 'bar',")
            html.append("                marker: {")
            html.append("                    color: 'rgba(66, 133, 244, 0.8)'")
            html.append("                }")
            html.append("            };")
            html.append("            ")
            html.append("            const layout = {")
            html.append("                title: 'WebGPU Execution Times by Model',")
            html.append("                xaxis: { title: 'Model' },")
            html.append("                yaxis: { title: 'Execution Time (ms)' }")
            html.append("            };")
            html.append("            ")
            html.append("            Plotly.newPlot('performance-chart', [trace], layout);")
            html.append("        } else {")
            html.append("            document.getElementById('performance-chart').innerHTML = '<p>No performance data available for visualization</p>';")
            html.append("        }")
            html.append("    </script>")
        
        # Browser-specific section
        if browser_specific_data:
            html.append(f"    <h2>{browser} WebGPU Performance</h2>")
            html.append(f"    <p>Found {len(browser_specific_data)} test results for {browser}.</p>")
            html.append("    <table>")
            html.append("        <tr><th>Model</th><th>Test Date</th><th>Success</th><th>Execution Time (ms)</th></tr>")
            for data in browser_specific_data[:10]:  # Show top 10 results
                status_icon = "<span class=\"success\">✅</span>" if data[2] else "<span class=\"failure\">❌</span>"
                html.append(f"        <tr><td>{data[0]}</td><td>{data[1]}</td><td>{status_icon}</td><td>{data[3] if data[3] is not None else 'N/A'}</td></tr>")
            html.append("    </table>")
        
        # Shader metrics section
        if shader_metrics:
            html.append("    <h2>Shader Compilation Metrics</h2>")
            html.append("    <table>")
            html.append("        <tr><th>Model</th><th>Compilation Time (ms)</th><th>Shader Count</th><th>Cache Hits</th><th>Test Date</th></tr>")
            for metric in shader_metrics[:10]:  # Show top 10 results
                html.append(f"        <tr><td>{metric[0]}</td><td>{metric[1] if metric[1] is not None else 'N/A'}</td>" +
                           f"<td>{metric[2] if metric[2] is not None else 'N/A'}</td>" +
                           f"<td>{metric[3] if metric[3] is not None else 'N/A'}</td><td>{metric[4]}</td></tr>")
            html.append("    </table>")
            
            # Add visualization div for shader metrics
            html.append("    <div id=\"shader-chart\" style=\"width:100%; height:400px;\"></div>")
            
            # Add JavaScript for shader chart
            html.append("    <script>")
            html.append("        // Prepare data for shader chart")
            html.append("        const shaderData = {")
            html.append("            models: " + json.dumps([metric[0] for metric in shader_metrics[:10] if metric[1] is not None]) + ",")
            html.append("            times: " + json.dumps([metric[1] for metric in shader_metrics[:10] if metric[1] is not None]) + ",")
            html.append("            counts: " + json.dumps([metric[2] for metric in shader_metrics[:10] if metric[2] is not None]) + "")
            html.append("        };")
            html.append("        ")
            html.append("        // Create chart")
            html.append("        if (shaderData.models.length > 0) {")
            html.append("            const trace1 = {")
            html.append("                x: shaderData.models,")
            html.append("                y: shaderData.times,")
            html.append("                name: 'Compilation Time (ms)',")
            html.append("                type: 'bar',")
            html.append("                marker: { color: 'rgba(66, 133, 244, 0.8)' }")
            html.append("            };")
            html.append("            ")
            html.append("            const trace2 = {")
            html.append("                x: shaderData.models,")
            html.append("                y: shaderData.counts,")
            html.append("                name: 'Shader Count',")
            html.append("                type: 'bar',")
            html.append("                marker: { color: 'rgba(219, 68, 55, 0.8)' }")
            html.append("            };")
            html.append("            ")
            html.append("            const layout = {")
            html.append("                title: 'Shader Compilation Metrics by Model',")
            html.append("                xaxis: { title: 'Model' },")
            html.append("                yaxis: { title: 'Value' },")
            html.append("                barmode: 'group'")
            html.append("            };")
            html.append("            ")
            html.append("            Plotly.newPlot('shader-chart', [trace1, trace2], layout);")
            html.append("        } else {")
            html.append("            document.getElementById('shader-chart').innerHTML = '<p>No shader data available for visualization</p>';")
            html.append("        }")
            html.append("    </script>")
        
        # Compute shader optimization section
        if compute_shader_data:
            html.append("    <h2>Compute Shader Optimization Analysis</h2>")
            html.append("    <table>")
            html.append("        <tr><th>Model</th><th>Optimization</th><th>Execution Time (ms)</th><th>Browser</th><th>Test Date</th></tr>")
            for data in compute_shader_data[:10]:  # Show top 10 results
                html.append(f"        <tr><td>{data[0]}</td><td>{data[1] if data[1] is not None else 'N/A'}</td>" +
                           f"<td>{data[2] if data[2] is not None else 'N/A'}</td>" +
                           f"<td>{data[3] if data[3] is not None else 'N/A'}</td><td>{data[4]}</td></tr>")
            html.append("    </table>")
        
        # WebGPU vs other hardware comparison
        if comparison_data:
            html.append("    <h2>WebGPU vs Other Hardware</h2>")
            
            # Group by model
            model_data = {}
            for data in comparison_data:
                model = data[0]
                if model not in model_data:
                    model_data[model] = []
                model_data[model].append(data)
            
            # Create a table for each model
            for model, data_points in model_data.items():
                html.append(f"    <h3>Model: {model}</h3>")
                html.append("    <table>")
                html.append("        <tr><th>Hardware</th><th>Avg Latency (ms)</th><th>Throughput (items/s)</th><th>Memory (MB)</th></tr>")
                for data in data_points:
                    html.append(f"        <tr><td>{data[1]}</td>" +
                               f"<td>{data[2]:.2f if data[2] is not None else 'N/A'}</td>" +
                               f"<td>{data[3]:.2f if data[3] is not None else 'N/A'}</td>" +
                               f"<td>{data[4]:.2f if data[4] is not None else 'N/A'}</td></tr>")
                html.append("    </table>")
                
                # Add visualization div for hardware comparison
                html.append(f"    <div id=\"hardware-chart-{model.replace(' ', '_')}\" style=\"width:100%; height:400px;\"></div>")
                
                # Add JavaScript for hardware comparison chart
                html.append("    <script>")
                html.append("        // Prepare data for hardware comparison chart")
                html.append("        const hardwareData_" + model.replace(' ', '_') + " = {")
                html.append("            hardware: " + json.dumps([d[1] for d in data_points if d[3] is not None]) + ",")
                html.append("            throughput: " + json.dumps([d[3] for d in data_points if d[3] is not None]) + "")
                html.append("        };")
                html.append("        ")
                html.append("        // Create chart")
                html.append("        if (hardwareData_" + model.replace(' ', '_') + ".hardware.length > 0) {")
                html.append("            const trace = {")
                html.append("                x: hardwareData_" + model.replace(' ', '_') + ".hardware,")
                html.append("                y: hardwareData_" + model.replace(' ', '_') + ".throughput,")
                html.append("                type: 'bar',")
                html.append("                marker: {")
                html.append("                    color: 'rgba(15, 157, 88, 0.8)'")
                html.append("                }")
                html.append("            };")
                html.append("            ")
                html.append("            const layout = {")
                html.append(f"                title: 'Throughput Comparison for {model}',")
                html.append("                xaxis: { title: 'Hardware' },")
                html.append("                yaxis: { title: 'Throughput (items/s)' }")
                html.append("            };")
                html.append("            ")
                html.append("            Plotly.newPlot('hardware-chart-" + model.replace(' ', '_') + "', [trace], layout);")
                html.append("        } else {")
                html.append("            document.getElementById('hardware-chart-" + model.replace(' ', '_') + "').innerHTML = '<p>No comparison data available for visualization</p>';")
                html.append("        }")
                html.append("    </script>")
        
        # Recommendations section
        html.append("    <h2>Recommendations</h2>")
        
        # Add general recommendations
        html.append("    <h3>General Recommendations</h3>")
        html.append("    <div class=\"recommendation\">")
        html.append("        <ul>")
        html.append("            <li>Use shader precompilation to improve initial load times</li>")
        html.append("            <li>Consider WebGPU for models that are compatible with browser environment</li>")
        html.append("            <li>Test with multiple browsers to optimize for specific user environments</li>")
        html.append("        </ul>")
        html.append("    </div>")
        
        # Add browser-specific recommendations if available
        if browser:
            html.append(f"    <h3>{browser}-Specific Recommendations</h3>")
            html.append("    <div class=\"recommendation\">")
            html.append("        <ul>")
            if browser.lower() == "chrome":
                html.append("            <li>Ensure latest Chrome version is used for best WebGPU performance</li>")
                html.append("            <li>For compute-heavy workloads, consider balanced workgroup sizes (e.g., 128x2x1)</li>")
            elif browser.lower() == "firefox":
                html.append("            <li>For audio models, Firefox often shows 20-30% better performance with compute shaders</li>")
                html.append("            <li>Use 256x1x1 workgroup size for best performance with audio models</li>")
            elif browser.lower() == "safari":
                html.append("            <li>WebGPU support in Safari has limitations; test thoroughly</li>")
                html.append("            <li>Fall back to WebNN for broader compatibility on Safari</li>")
            elif browser.lower() == "edge":
                html.append("            <li>Edge performs similarly to Chrome for most WebGPU workloads</li>")
                html.append("            <li>Consider enabling hardware acceleration in browser settings</li>")
            html.append("        </ul>")
            html.append("    </div>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)

    def _generate_webgpu_json_report(self, webgpu_data, browser_specific_data, 
                                   shader_metrics, compute_shader_data, comparison_data, browser=None):
        """Generate a JSON WebGPU analysis report."""
        # Convert database tuples to dictionaries for JSON serialization
        report = {
            "generated_at": datetime.now().isoformat(),
            "browser": browser
        }
        
        # Add WebGPU performance data
        if webgpu_data:
            report["webgpu_data"] = [
                {
                    "model": data[0],
                    "test_date": str(data[1]),
                    "success": bool(data[2]),
                    "execution_time_ms": data[3]
                }
                for data in webgpu_data
            ]
            
            # Calculate success metrics
            success_count = sum(1 for data in webgpu_data if data[2])
            report["success_metrics"] = {
                "total_tests": len(webgpu_data),
                "successful_tests": success_count,
                "success_rate": (success_count / len(webgpu_data) * 100) if webgpu_data else 0
            }
        
        # Add other sections
        if browser_specific_data:
            report["browser_specific_data"] = [
                {
                    "model": data[0],
                    "test_date": str(data[1]),
                    "success": bool(data[2]),
                    "execution_time_ms": data[3]
                }
                for data in browser_specific_data
            ]
        
        if shader_metrics:
            report["shader_metrics"] = [
                {
                    "model": metric[0],
                    "compilation_time_ms": metric[1],
                    "shader_count": metric[2],
                    "cache_hits": metric[3],
                    "test_date": str(metric[4])
                }
                for metric in shader_metrics
            ]
        
        if compute_shader_data:
            report["compute_shader_data"] = [
                {
                    "model": data[0],
                    "optimization": data[1],
                    "execution_time_ms": data[2],
                    "browser": data[3],
                    "test_date": str(data[4])
                }
                for data in compute_shader_data
            ]
        
        if comparison_data:
            # Group by model
            comparison_by_model = {}
            for data in comparison_data:
                model = data[0]
                if model not in comparison_by_model:
                    comparison_by_model[model] = []
                
                comparison_by_model[model].append({
                    "hardware": data[1],
                    "avg_latency_ms": data[2],
                    "throughput_items_per_second": data[3],
                    "memory_mb": data[4]
                })
            
            report["hardware_comparison"] = comparison_by_model
        
        # Add recommendations
        report["recommendations"] = {
            "general": [
                "Use shader precompilation to improve initial load times",
                "Consider WebGPU for models that are compatible with browser environment",
                "Test with multiple browsers to optimize for specific user environments"
            ]
        }
        
        if browser:
            report["recommendations"][f"{browser.lower()}_specific"] = []
            if browser.lower() == "chrome":
                report["recommendations"][f"{browser.lower()}_specific"] = [
                    "Ensure latest Chrome version is used for best WebGPU performance",
                    "For compute-heavy workloads, consider balanced workgroup sizes (e.g., 128x2x1)"
                ]
            elif browser.lower() == "firefox":
                report["recommendations"][f"{browser.lower()}_specific"] = [
                    "For audio models, Firefox often shows 20-30% better performance with compute shaders",
                    "Use 256x1x1 workgroup size for best performance with audio models"
                ]
        
        return json.dumps(report, indent=2)


class QualcommTestHandler:
    """
    Handler for testing models on Qualcomm AI Engine.
    
    This class provides methods for:
    1. Detecting Qualcomm hardware and SDK
    2. Converting models to Qualcomm formats (QNN or DLC)
    3. Running inference on Qualcomm hardware
    4. Measuring power consumption and thermal metrics
    """
    
    def __init__(self):
        """Initialize the Qualcomm test handler."""
        self.has_qualcomm = False
        self.sdk_type = None  # 'QNN' or 'QTI'
        self.sdk_version = None
        self.device_name = None
        self.mock_mode = False
        
        # Detect Qualcomm SDK and capabilities
        self._detect_qualcomm()
    
    def _detect_qualcomm(self):
        """Detect Qualcomm hardware and SDK."""
        # Check if Qualcomm SDK is available (QNN or QTI SDK)
        try:
            # First try QNN SDK
            if importlib.util.find_spec("qnn_wrapper") is not None:
                self.has_qualcomm = True
                self.sdk_type = "QNN"
                
                # Try to get SDK version
                try:
                    import qnn_wrapper
                    self.sdk_version = getattr(qnn_wrapper, "__version__", "unknown")
                except (ImportError, AttributeError):
                    self.sdk_version = "unknown"
                    
                print(f"Detected Qualcomm QNN SDK version {self.sdk_version}")
                return
                
            # Try QTI SDK
            if importlib.util.find_spec("qti") is not None:
                self.has_qualcomm = True
                self.sdk_type = "QTI"
                
                # Try to get SDK version
                try:
                    import qti
                    self.sdk_version = getattr(qti, "__version__", "unknown")
                except (ImportError, AttributeError):
                    self.sdk_version = "unknown"
                    
                print(f"Detected Qualcomm QTI SDK version {self.sdk_version}")
                return
                
            # Check for environment variable as fallback
            if os.environ.get("QUALCOMM_SDK"):
                self.has_qualcomm = True
                self.sdk_type = os.environ.get("QUALCOMM_SDK_TYPE", "QNN")
                self.sdk_version = os.environ.get("QUALCOMM_SDK_VERSION", "unknown")
                self.mock_mode = True
                print(f"Using Qualcomm {self.sdk_type} SDK from environment variables (mock mode)")
                return
                
            # No Qualcomm SDK detected
            self.has_qualcomm = False
            print("No Qualcomm AI Engine SDK detected")
            
        except Exception as e:
            # Error during detection
            print(f"Error detecting Qualcomm SDK: {e}")
            self.has_qualcomm = False
    
    def is_available(self):
        """Check if Qualcomm AI Engine is available."""
        return self.has_qualcomm
    
    def get_device_info(self):
        """Get information about the Qualcomm device."""
        if not self.has_qualcomm:
            return {"error": "Qualcomm AI Engine not available"}
            
        device_info = {
            "sdk_type": self.sdk_type,
            "sdk_version": self.sdk_version,
            "device_name": self.device_name or "unknown",
            "mock_mode": self.mock_mode,
            "has_power_metrics": self._has_power_metrics()
        }
        
        # Try to get additional device information when available
        if self.sdk_type == "QNN" and not self.mock_mode:
            try:
                import qnn_wrapper
                # Add QNN-specific device information
                if hasattr(qnn_wrapper, "get_device_info"):
                    qnn_info = qnn_wrapper.get_device_info()
                    device_info.update(qnn_info)
            except (ImportError, AttributeError, Exception) as e:
                device_info["error"] = f"Error getting QNN device info: {e}"
                
        elif self.sdk_type == "QTI" and not self.mock_mode:
            try:
                import qti
                # Add QTI-specific device information
                if hasattr(qti, "get_device_info"):
                    qti_info = qti.get_device_info()
                    device_info.update(qti_info)
            except (ImportError, AttributeError, Exception) as e:
                device_info["error"] = f"Error getting QTI device info: {e}"
                
        return device_info
    
    def _has_power_metrics(self):
        """Check if power consumption metrics are available."""
        if self.mock_mode:
            # Mock mode always reports power metrics as available
            return True
            
        # Real implementation needs to check if power metrics APIs are available
        if self.sdk_type == "QNN":
            try:
                import qnn_wrapper
                return hasattr(qnn_wrapper, "get_power_metrics") or hasattr(qnn_wrapper, "monitor_power")
            except (ImportError, AttributeError):
                return False
                
        elif self.sdk_type == "QTI":
            try:
                import qti
                return hasattr(qti.aisw, "power_metrics") or hasattr(qti, "monitor_power")
            except (ImportError, AttributeError):
                return False
                
        return False
    
    def convert_model(self, model_path, output_path, model_type="bert"):
        """
        Convert a model to Qualcomm format (QNN or DLC).
        
        Args:
            model_path: Path to input model (ONNX or PyTorch)
            output_path: Path for converted model
            model_type: Type of model (bert, llm, vision, etc.)
            
        Returns:
            dict: Conversion results
        """
        if not self.has_qualcomm:
            return {"error": "Qualcomm AI Engine not available"}
            
        # Mock implementation for testing
        if self.mock_mode:
            print(f"Mock Qualcomm: Converting {model_path} to {output_path}")
            return {
                "status": "success",
                "input_path": model_path,
                "output_path": output_path,
                "model_type": model_type,
                "sdk_type": self.sdk_type,
                "mock_mode": True
            }
            
        # Real implementation based on SDK type
        try:
            if self.sdk_type == "QNN":
                return self._convert_model_qnn(model_path, output_path, model_type)
            elif self.sdk_type == "QTI":
                return self._convert_model_qti(model_path, output_path, model_type)
            else:
                return {"error": f"Unsupported SDK type: {self.sdk_type}"}
        except Exception as e:
            return {
                "error": f"Error converting model: {e}",
                "traceback": traceback.format_exc()
            }
    
    def _convert_model_qnn(self, model_path, output_path, model_type):
        """Convert model using QNN SDK."""
        import qnn_wrapper
        
        # Set conversion parameters based on model type
        params = {
            "input_model": model_path,
            "output_model": output_path,
            "model_type": model_type
        }
        
        # Add model-specific parameters
        if model_type == "bert":
            params["optimization_level"] = "performance"
        elif model_type == "llm":
            params["quantization"] = True
        elif model_type in ["vision", "clip"]:
            params["input_layout"] = "NCHW"
            
        # Convert model
        result = qnn_wrapper.convert_model(**params)
        
        return {
            "status": "success" if result else "failure",
            "input_path": model_path,
            "output_path": output_path,
            "model_type": model_type,
            "sdk_type": "QNN",
            "params": params
        }
    
    def _convert_model_qti(self, model_path, output_path, model_type):
        """Convert model using QTI SDK."""
        from qti.aisw import dlc_utils
        
        # Set conversion parameters based on model type
        params = {
            "input_model": model_path,
            "output_model": output_path,
            "model_type": model_type
        }
        
        # Add model-specific parameters
        if model_type == "bert":
            params["optimization_level"] = "performance"
        elif model_type == "llm":
            params["quantization"] = True
        elif model_type in ["vision", "clip"]:
            params["input_layout"] = "NCHW"
            
        # Convert model
        result = dlc_utils.convert_onnx_to_dlc(**params)
        
        return {
            "status": "success" if result else "failure",
            "input_path": model_path,
            "output_path": output_path,
            "model_type": model_type,
            "sdk_type": "QTI",
            "params": params
        }
    
    def run_inference(self, model_path, input_data, monitor_metrics=True, model_type=None):
        """
        Run inference on Qualcomm hardware.
        
        Args:
            model_path: Path to converted model
            input_data: Input data for inference
            monitor_metrics: Whether to monitor power and thermal metrics
            model_type: Type of model (vision, text, audio, llm) for more accurate power profiling
            
        Returns:
            dict: Inference results with metrics
        """
        if not self.has_qualcomm:
            return {"error": "Qualcomm AI Engine not available"}
            
        # Determine model type if not provided
        if model_type is None:
            model_type = self._infer_model_type(model_path, input_data)
            
        # Mock implementation for testing
        if self.mock_mode:
            print(f"Mock Qualcomm: Running inference on {model_path} (type: {model_type})")
            
            # Generate mock results based on model type
            import numpy as np
            
            # Output shape depends on model type
            if model_type == "vision":
                mock_output = np.random.randn(1, 1000)  # Classification logits
            elif model_type == "text":
                mock_output = np.random.randn(1, 768)  # Embedding vector
            elif model_type == "audio":
                mock_output = np.random.randn(1, 128, 20)  # Audio features
            elif model_type == "llm":
                # Generate a small token sequence
                mock_output = np.random.randint(0, 50000, size=(1, 10))
            else:
                mock_output = np.random.randn(1, 768)  # Default embedding
            
            # Get power monitoring data with model type
            metrics_data = self._start_metrics_monitoring(model_type)
            
            # Simulate processing time based on model type
            if model_type == "llm":
                time.sleep(0.05)  # LLMs are slower
            elif model_type == "vision":
                time.sleep(0.02)  # Vision models moderately fast
            elif model_type == "audio":
                time.sleep(0.03)  # Audio processing moderate
            else:
                time.sleep(0.01)  # Text embeddings are fast
                
            # Generate metrics with model-specific characteristics
            metrics = self._stop_metrics_monitoring(metrics_data)
            
            # Include device info in the result
            device_info = {
                "device_name": "Mock Qualcomm Device",
                "sdk_type": self.sdk_type,
                "sdk_version": self.sdk_version or "unknown",
                "mock_mode": self.mock_mode,
                "has_power_metrics": True,
                "model_type": model_type
            }
            
            # Add throughput metric based on model type
            throughput_map = {
                "vision": {"units": "images/second", "value": 30.0},
                "text": {"units": "samples/second", "value": 80.0},
                "audio": {"units": "seconds of audio/second", "value": 5.0},
                "llm": {"units": "tokens/second", "value": 15.0},
                "generic": {"units": "samples/second", "value": 40.0}
            }
            
            throughput_info = throughput_map.get(model_type, throughput_map["generic"])
            
            return {
                "status": "success",
                "output": mock_output,
                "metrics": metrics,
                "device_info": device_info,
                "sdk_type": self.sdk_type,
                "model_type": model_type,
                "throughput": throughput_info["value"],
                "throughput_units": throughput_info["units"]
            }
            
        # Real implementation based on SDK type
        try:
            metrics_data = {}
            if monitor_metrics and self._has_power_metrics():
                # Start metrics monitoring with model type
                metrics_data = self._start_metrics_monitoring(model_type)
                
            # Run inference
            if self.sdk_type == "QNN":
                result = self._run_inference_qnn(model_path, input_data)
            elif self.sdk_type == "QTI":
                result = self._run_inference_qti(model_path, input_data)
            else:
                return {"error": f"Unsupported SDK type: {self.sdk_type}"}
                
            # Add model type to result
            result["model_type"] = model_type
                
            # Stop metrics monitoring and update result
            if monitor_metrics and self._has_power_metrics():
                metrics = self._stop_metrics_monitoring(metrics_data)
                result["metrics"] = metrics
            
            # Always include device info in the result
            device_info = self.get_device_info()
            device_info["model_type"] = model_type  # Include model type in device info
            result["device_info"] = device_info
                
            return result
            
        except Exception as e:
            return {
                "error": f"Error running inference: {e}",
                "traceback": traceback.format_exc()
            }
            
    def _infer_model_type(self, model_path, input_data):
        """Infer model type from model path and input data."""
        model_path = str(model_path).lower()
        
        # Check model path for indicators
        if any(x in model_path for x in ["vit", "clip", "vision", "image", "resnet", "detr", "vgg"]):
            return "vision"
        elif any(x in model_path for x in ["whisper", "wav2vec", "clap", "audio", "speech", "voice"]):
            return "audio"
        elif any(x in model_path for x in ["llava", "llama", "gpt", "llm", "falcon", "mistral", "phi"]):
            return "llm"
        elif any(x in model_path for x in ["bert", "roberta", "text", "embed", "sentence", "bge"]):
            return "text"
            
        # Check input shape if input is numpy array
        if hasattr(input_data, "shape"):
            # Vision inputs often have 4 dimensions (batch, channels, height, width)
            if len(input_data.shape) == 4 and input_data.shape[1] in [1, 3]:
                return "vision"
            # Audio inputs typically have 2-3 dimensions
            elif len(input_data.shape) == 2 and input_data.shape[1] > 1000:  # Long sequence for audio
                return "audio"
            
        # Default to generic text model if no indicators found
        return "text"
    
    def _run_inference_qnn(self, model_path, input_data):
        """Run inference using QNN SDK."""
        import qnn_wrapper
        
        # Load model
        model = qnn_wrapper.QnnModel(model_path)
        
        # Record start time
        start_time = time.time()
        
        # Run inference
        output = model.execute(input_data)
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "status": "success",
            "output": output,
            "execution_time_ms": execution_time,
            "sdk_type": "QNN"
        }
    
    def _run_inference_qti(self, model_path, input_data):
        """Run inference using QTI SDK."""
        from qti.aisw.dlc_runner import DlcRunner
        
        # Load model
        model = DlcRunner(model_path)
        
        # Record start time
        start_time = time.time()
        
        # Run inference
        output = model.execute(input_data)
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "status": "success",
            "output": output,
            "execution_time_ms": execution_time,
            "sdk_type": "QTI"
        }
    
    def _start_metrics_monitoring(self, model_type=None):
        """
        Start monitoring power and thermal metrics.
        
        Args:
            model_type (str, optional): Type of model being benchmarked (vision, text, audio, llm).
                                        Used for more accurate power profiling.
        """
        metrics_data = {"start_time": time.time()}
        
        # Store model type for more accurate metrics later
        if model_type:
            metrics_data["model_type"] = model_type
        
        if self.mock_mode:
            return metrics_data
        
        # Real implementation based on SDK type
        if self.sdk_type == "QNN":
            try:
                import qnn_wrapper
                if hasattr(qnn_wrapper, "start_power_monitoring"):
                    # Pass model type if the SDK supports it
                    if hasattr(qnn_wrapper.start_power_monitoring, "__code__") and "model_type" in qnn_wrapper.start_power_monitoring.__code__.co_varnames:
                        metrics_data["monitor_handle"] = qnn_wrapper.start_power_monitoring(model_type=model_type)
                    else:
                        metrics_data["monitor_handle"] = qnn_wrapper.start_power_monitoring()
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not start QNN power monitoring: {e}")
                
        elif self.sdk_type == "QTI":
            try:
                import qti
                if hasattr(qti.aisw, "start_power_monitoring"):
                    # Pass model type if the SDK supports it
                    if hasattr(qti.aisw.start_power_monitoring, "__code__") and "model_type" in qti.aisw.start_power_monitoring.__code__.co_varnames:
                        metrics_data["monitor_handle"] = qti.aisw.start_power_monitoring(model_type=model_type)
                    else:
                        metrics_data["monitor_handle"] = qti.aisw.start_power_monitoring()
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not start QTI power monitoring: {e}")
                
        return metrics_data
    
    def _stop_metrics_monitoring(self, metrics_data):
        """Stop monitoring and collect metrics."""
        if self.mock_mode:
            # Generate more realistic mock metrics with improved metrics that match the schema
            elapsed_time = time.time() - metrics_data["start_time"]
            
            # Model-specific power profiles for different device types
            # Base power consumption varies by model type to simulate realistic device behavior
            model_type = metrics_data.get("model_type", "generic")
            
            # Base power values by model type (in milliwatts)
            power_profiles = {
                "vision": {"base": 500.0, "variance": 60.0, "peak_factor": 1.3, "idle_factor": 0.35},
                "text": {"base": 400.0, "variance": 40.0, "peak_factor": 1.2, "idle_factor": 0.4},
                "audio": {"base": 550.0, "variance": 70.0, "peak_factor": 1.35, "idle_factor": 0.3},
                "llm": {"base": 650.0, "variance": 100.0, "peak_factor": 1.4, "idle_factor": 0.25},
                "generic": {"base": 450.0, "variance": 50.0, "peak_factor": 1.25, "idle_factor": 0.4}
            }
            
            # Get profile for this model type
            profile = power_profiles.get(model_type, power_profiles["generic"])
            
            # Base power consumption (randomized slightly for variance)
            base_power = profile["base"] + float(numpy.random.rand() * profile["variance"])
            
            # Peak power is higher than base
            peak_power = base_power * profile["peak_factor"] * (1.0 + float(numpy.random.rand() * 0.1))
            
            # Idle power is lower than base
            idle_power = base_power * profile["idle_factor"] * (1.0 + float(numpy.random.rand() * 0.05))
            
            # Average power calculation - weighted average that accounts for computation phases
            # Typically devices spend ~60% at base power, 15% at peak, and 25% at lower power
            avg_power = (base_power * 0.6) + (peak_power * 0.15) + ((base_power * 0.7) * 0.25)
            
            # Energy is power * time
            energy = avg_power * elapsed_time
            
            # Realistic temperature for mobile SoC under load - varies by model type
            base_temp = 37.0 + (model_type == "llm") * 3.0 + (model_type == "vision") * 1.0
            temp_variance = 6.0 + (model_type == "llm") * 2.0
            temperature = base_temp + float(numpy.random.rand() * temp_variance)
            
            # Thermal throttling detection (simulated when temperature is very high)
            thermal_throttling = temperature > 45.0
            
            # Power efficiency metric (tokens or samples per joule)
            # This is an important metric for mobile devices
            throughput = 25.0  # tokens/second or samples/second (model dependent)
            energy_efficiency = (throughput * elapsed_time) / (energy / 1000.0)  # items per joule
            
            # Battery impact (estimated percentage of battery used per hour at this rate)
            # Assuming a typical mobile device with 3000 mAh battery at 3.7V (~40,000 joules)
            hourly_energy = energy * (3600.0 / elapsed_time)  # mJ used per hour
            battery_impact_hourly = (hourly_energy / 40000000.0) * 100.0  # percentage of battery per hour
            
            return {
                "power_consumption_mw": base_power,
                "energy_consumption_mj": energy,
                "temperature_celsius": temperature,
                "monitoring_duration_ms": elapsed_time * 1000,
                "average_power_mw": avg_power,
                "peak_power_mw": peak_power,
                "idle_power_mw": idle_power,
                "execution_time_ms": elapsed_time * 1000,
                "energy_efficiency_items_per_joule": energy_efficiency,
                "thermal_throttling_detected": thermal_throttling,
                "battery_impact_percent_per_hour": battery_impact_hourly,
                "model_type": model_type,
                "mock_mode": True
            }
            
        # For real hardware, calculate metrics and ensure complete set of fields
        elapsed_time = time.time() - metrics_data["start_time"]
        
        # Initialize with default metrics
        metrics = {
            "monitoring_duration_ms": elapsed_time * 1000
        }
        
        # Real implementation based on SDK type
        try:
            if self.sdk_type == "QNN" and "monitor_handle" in metrics_data:
                try:
                    import qnn_wrapper
                    if hasattr(qnn_wrapper, "stop_power_monitoring"):
                        power_metrics = qnn_wrapper.stop_power_monitoring(metrics_data["monitor_handle"])
                        metrics.update(power_metrics)
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not stop QNN power monitoring: {e}")
                    
            elif self.sdk_type == "QTI" and "monitor_handle" in metrics_data:
                try:
                    import qti
                    if hasattr(qti.aisw, "stop_power_monitoring"):
                        power_metrics = qti.aisw.stop_power_monitoring(metrics_data["monitor_handle"])
                        metrics.update(power_metrics)
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not stop QTI power monitoring: {e}")
            
            # Check for missing essential metrics and calculate them if possible
            if "power_consumption_mw" in metrics and "monitoring_duration_ms" in metrics:
                # Calculate energy if not provided
                if "energy_consumption_mj" not in metrics:
                    metrics["energy_consumption_mj"] = metrics["power_consumption_mw"] * (metrics["monitoring_duration_ms"] / 1000.0)
                
                # Use power consumption as average if not provided
                if "average_power_mw" not in metrics:
                    metrics["average_power_mw"] = metrics["power_consumption_mw"]
                
                # Estimate peak power if not provided (typically 20% higher than average)
                if "peak_power_mw" not in metrics and "average_power_mw" in metrics:
                    metrics["peak_power_mw"] = metrics["average_power_mw"] * 1.2
                
                # Estimate idle power if not provided (typically 40% of average)
                if "idle_power_mw" not in metrics and "average_power_mw" in metrics:
                    metrics["idle_power_mw"] = metrics["average_power_mw"] * 0.4
                
                # Make sure execution time is included
                if "execution_time_ms" not in metrics:
                    metrics["execution_time_ms"] = metrics["monitoring_duration_ms"]
                
                # Add energy efficiency metrics (if we can estimate throughput)
                if "throughput" in metrics and "energy_consumption_mj" in metrics and metrics["energy_consumption_mj"] > 0:
                    # Calculate items processed
                    items_processed = metrics["throughput"] * (metrics["monitoring_duration_ms"] / 1000.0)
                    # Calculate energy efficiency (items per joule)
                    metrics["energy_efficiency_items_per_joule"] = items_processed / (metrics["energy_consumption_mj"] / 1000.0)
                
                # Detect thermal throttling based on temperature
                if "temperature_celsius" in metrics:
                    # Thermal throttling typically occurs around 80°C for mobile devices
                    metrics["thermal_throttling_detected"] = metrics["temperature_celsius"] > 80.0
                    
                # Estimate battery impact (for mobile devices)
                if "energy_consumption_mj" in metrics:
                    # Estimate hourly energy consumption
                    hourly_energy = metrics["energy_consumption_mj"] * (3600.0 / (metrics["monitoring_duration_ms"] / 1000.0))
                    # Assuming a typical mobile device with 3000 mAh battery at 3.7V (~40,000 joules)
                    metrics["battery_impact_percent_per_hour"] = (hourly_energy / 40000000.0) * 100.0
                
        except Exception as e:
            print(f"Warning: Error calculating power metrics: {e}")
            
        return metrics

class test_ipfs_accelerate:
    """
    Test class for IPFS Accelerate Python Framework.
    
    This class provides methods to test the IPFS Accelerate Python framework and its components:
    1. Hardware backend testing
    2. IPFS accelerate model endpoint testing
    3. Local endpoints (CUDA, OpenVINO, CPU)
    4. API endpoints (TEI, OVMS)
    5. Network endpoints (libp2p, WebNN)
    
    The test process follows these phases:
    - Phase 1: Test with models defined in global metadata
    - Phase 2: Test with models from mapped_models.json
    - Phase 3: Collect and analyze test results
    - Phase 4: Generate test reports
    """
    
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the test_ipfs_accelerate class.
        
        Args:
            resources (dict, optional): Dictionary containing resources like endpoints. Defaults to None.
            metadata (dict, optional): Dictionary containing metadata like models list. Defaults to None.
        """
        # Initialize resources
        if resources is None:
            self.resources = {}
        else:
            self.resources = resources
        
        # Initialize metadata
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        
        # Initialize ipfs_accelerate_py
        if "ipfs_accelerate_py" not in dir(self):
            if "ipfs_accelerate_py" not in list(self.resources.keys()):
                try:
                    from ipfs_accelerate_py import ipfs_accelerate_py
                    self.resources["ipfs_accelerate_py"] = ipfs_accelerate_py(resources, metadata)
                    self.ipfs_accelerate_py = self.resources["ipfs_accelerate_py"]
                except Exception as e:
                    print(f"Error initializing ipfs_accelerate_py: {str(e)}")
                    print(traceback.format_exc())
                    self.resources["ipfs_accelerate_py"] = None
                    self.ipfs_accelerate_py = None
            else:
                self.ipfs_accelerate_py = self.resources["ipfs_accelerate_py"]
        
        # Initialize test_hardware_backend
        if "test_hardware_backend" not in dir(self):
            if "test_hardware_backend" not in list(self.resources.keys()):
                try:
                    from test_hardware_backend import test_hardware_backend
                    self.resources["test_backend"] = test_hardware_backend(resources, metadata)
                    self.test_backend = self.resources["test_backend"]
                except Exception as e:
                    print(f"Error initializing test_hardware_backend: {str(e)}")
                    print(traceback.format_exc())
                    self.resources["test_backend"] = None
                    self.test_backend = None
            else:
                self.test_backend = self.resources["test_backend"]
        
        # Initialize test_api_backend
        if "test_api_backend" not in dir(self):
            if "test_api_backend" not in list(self.resources.keys()):
                try:
                    from test_api_backend import test_api_backend
                    self.resources["test_api_backend"] = test_api_backend(resources, metadata)
                    self.test_api_backend = self.resources["test_api_backend"]
                except Exception as e:
                    print(f"Error initializing test_api_backend: {str(e)}")
                    print(traceback.format_exc())
                    self.resources["test_api_backend"] = None
                    self.test_api_backend = None
            else:
                self.test_api_backend = self.resources["test_api_backend"]
        
        # Initialize torch
        if "torch" not in dir(self):
            if "torch" not in list(self.resources.keys()):
                try:
                    import torch
                    self.resources["torch"] = torch
                    self.torch = self.resources["torch"]
                except Exception as e:
                    print(f"Error importing torch: {str(e)}")
                    self.resources["torch"] = None
                    self.torch = None
            else:
                self.torch = self.resources["torch"]
                
        # Initialize transformers module - needed for most skill tests
        if "transformers" not in list(self.resources.keys()):
            try:
                import transformers
                self.resources["transformers"] = transformers
                print("  Added transformers module to resources")
            except Exception as e:
                print(f"Error importing transformers: {str(e)}")
                # Create MagicMock for transformers if import fails
                try:
                    from unittest.mock import MagicMock
                    self.resources["transformers"] = MagicMock()
                    print("  Added MagicMock for transformers to resources")
                except Exception as mock_error:
                    print(f"Error creating MagicMock for transformers: {str(mock_error)}")
                    self.resources["transformers"] = None
        
        # Ensure required resource dictionaries exist and are properly structured
        required_resource_keys = [
            "local_endpoints", 
            "openvino_endpoints", 
            "tokenizer"
        ]
        
        for key in required_resource_keys:
            if key not in self.resources:
                self.resources[key] = {}
        
        # Convert list structures to dictionary structures if needed
        self._convert_resource_structures()
        
        return None
        
    def _convert_resource_structures(self):
        """
        Convert list-based resources to dictionary-based structures.
        
        This method ensures all resources use the proper dictionary structure expected by
        ipfs_accelerate_py.init_endpoints. It handles conversion of:
        - local_endpoints: from list to nested dictionary
        - tokenizer: from list to nested dictionary
        """
        # Convert local_endpoints from list to dictionary if needed
        if isinstance(self.resources.get("local_endpoints"), list) and self.resources["local_endpoints"]:
            local_endpoints_dict = {}
            
            # Convert list entries to dictionary structure
            for endpoint_entry in self.resources["local_endpoints"]:
                if len(endpoint_entry) >= 2:
                    model = endpoint_entry[0]
                    endpoint_type = endpoint_entry[1]
                    
                    # Create nested structure
                    if model not in local_endpoints_dict:
                        local_endpoints_dict[model] = []
                    
                    # Add endpoint entry to the model's list
                    local_endpoints_dict[model].append(endpoint_entry)
            
            # Replace list with dictionary
            self.resources["local_endpoints"] = local_endpoints_dict
            print(f"  Converted local_endpoints from list to dictionary with {len(local_endpoints_dict)} models")
        
        # Convert tokenizer from list to dictionary if needed
        if isinstance(self.resources.get("tokenizer"), list) and self.resources["tokenizer"]:
            tokenizer_dict = {}
            
            # Convert list entries to dictionary structure
            for tokenizer_entry in self.resources["tokenizer"]:
                if len(tokenizer_entry) >= 2:
                    model = tokenizer_entry[0]
                    endpoint_type = tokenizer_entry[1]
                    
                    # Create nested structure
                    if model not in tokenizer_dict:
                        tokenizer_dict[model] = {}
                    
                    # Initialize with None, will be filled during endpoint creation
                    tokenizer_dict[model][endpoint_type] = None
            
            # Replace list with dictionary
            self.resources["tokenizer"] = tokenizer_dict
            print(f"  Converted tokenizer from list to dictionary with {len(tokenizer_dict)} models")
        
        # Add endpoint_handler dictionary if it doesn't exist
        if "endpoint_handler" not in self.resources:
            self.resources["endpoint_handler"] = {}
            
        # Ensure proper structure for endpoint_handler
        for model in self.resources.get("local_endpoints", {}):
            if model not in self.resources["endpoint_handler"]:
                self.resources["endpoint_handler"][model] = {}
                
        # Create structured resources dictionary for ipfs_accelerate_py.init_endpoints
        if "structured_resources" not in self.resources:
            self.resources["structured_resources"] = {
                "tokenizer": self.resources.get("tokenizer", {}),
                "endpoint_handler": self.resources.get("endpoint_handler", {}),
                "endpoints": {
                    "local_endpoints": self.resources.get("local_endpoints", {}),
                    "api_endpoints": self.resources.get("tei_endpoints", {})
                }
            }
    
    async def get_huggingface_model_types(self):
        """
        Get a list of all Hugging Face model types.
        
        Returns:
            list: Sorted list of model types
        """
        # Initialize transformers if not already done
        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                try:
                    import transformers
                    self.resources["transformers"] = transformers
                    self.transformers = self.resources["transformers"]
                except Exception as e:
                    print(f"Error importing transformers: {str(e)}")
                    return []
            else:
                self.transformers = self.resources["transformers"]

        try:
            # Get all model types from the MODEL_MAPPING
            model_types = []
            for config in self.transformers.MODEL_MAPPING.keys():
                if hasattr(config, 'model_type'):
                    model_types.append(config.model_type)

            # Add model types from the AutoModel registry
            model_types.extend(list(self.transformers.MODEL_MAPPING._model_mapping.keys()))
            
            # Remove duplicates and sort
            model_types = sorted(list(set(model_types)))
            return model_types
        except Exception as e:
            print(f"Error getting Hugging Face model types: {str(e)}")
            print(traceback.format_exc())
            return []    
    
    def get_model_type(self, model_name=None, model_type=None):
        """
        Get the model type for a given model name.
        
        Args:
            model_name (str, optional): The model name. Defaults to None.
            model_type (str, optional): The model type. Defaults to None.
            
        Returns:
            str: The model type
        """
        # Initialize transformers if not already done
        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                try:
                    import transformers
                    self.resources["transformers"] = transformers
                    self.transformers = self.resources["transformers"]
                except Exception as e:
                    print(f"Error importing transformers: {str(e)}")
                    return None
            else:
                self.transformers = self.resources["transformers"]

        # Get model type based on model name
        if model_name is not None:
            try:
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
            except Exception as e:
                print(f"Error getting model type for {model_name}: {str(e)}")
        
        return model_type
    
    async def test(self):
        """
        Main test method that tests both hardware backend and IPFS accelerate endpoints.
        
        This method performs the following tests:
        1. Test hardware backend with the models defined in metadata
        2. Test IPFS accelerate endpoints for both CUDA and OpenVINO platforms
        
        Returns:
            dict: Dictionary containing test results for hardware backend and IPFS accelerate
        """
        test_results = {
            "timestamp": str(datetime.now()),
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Running"
        }
        
        # Test hardware backend
        try:
            print("Testing hardware backend...")
            if self.test_backend is None:
                raise ValueError("test_backend is not initialized")
                
            if not hasattr(self.test_backend, "__test__") or not callable(self.test_backend.__test__):
                raise AttributeError("test_backend.__test__ method is not defined or not callable")
                
            # Check if test_backend exists and has __test__ method
            if not hasattr(self.test_backend, "__test__"):
                raise AttributeError("test_backend does not have __test__ method")
                
            # Check the signature of __test__ method to determine how to call it
            import inspect
            sig = inspect.signature(self.test_backend.__test__)
            param_count = len(sig.parameters)
            
            print(f"TestHardwareBackend.__test__() has {param_count} parameters: {list(sig.parameters.keys())}")
            
            # Call with appropriate number of parameters, handling various signature formats
            if asyncio.iscoroutinefunction(self.test_backend.__test__):
                # Handle async method
                if param_count == 1:  # Just self
                    test_results["test_backend"] = await self.test_backend.__test__()
                elif param_count == 2:  # Could be (self, resources) or (self, metadata)
                    # Check parameter names to determine what to pass
                    param_names = list(sig.parameters.keys())
                    if 'resources' in param_names:
                        test_results["test_backend"] = await self.test_backend.__test__(self.resources)
                    elif 'metadata' in param_names:
                        test_results["test_backend"] = await self.test_backend.__test__(self.metadata)
                    else:
                        # If parameter names aren't resources or metadata, try resources as default
                        test_results["test_backend"] = await self.test_backend.__test__(self.resources)
                elif param_count == 3:  # self, resources, metadata
                    test_results["test_backend"] = await self.test_backend.__test__(self.resources, self.metadata)
                else:
                    # For any other parameter count, attempt to call without params as fallback
                    print(f"Warning: Unexpected parameter count {param_count} for test_backend.__test__")
                    print(f"Attempting to call with no parameters as fallback")
                    test_results["test_backend"] = await self.test_backend.__test__()
            else:
                # Handle sync method
                if param_count == 1:  # Just self
                    test_results["test_backend"] = self.test_backend.__test__()
                elif param_count == 2:  # Could be (self, resources) or (self, metadata)
                    # Check parameter names to determine what to pass
                    param_names = list(sig.parameters.keys())
                    if 'resources' in param_names:
                        test_results["test_backend"] = self.test_backend.__test__(self.resources)
                    elif 'metadata' in param_names:
                        test_results["test_backend"] = self.test_backend.__test__(self.metadata)
                    else:
                        # If parameter names aren't resources or metadata, try resources as default
                        test_results["test_backend"] = self.test_backend.__test__(self.resources)
                elif param_count == 3:  # self, resources, metadata
                    test_results["test_backend"] = self.test_backend.__test__(self.resources, self.metadata)
                else:
                    # For any other parameter count, attempt to call without params as fallback
                    print(f"Warning: Unexpected parameter count {param_count} for test_backend.__test__")
                    print(f"Attempting to call with no parameters as fallback")
                    test_results["test_backend"] = self.test_backend.__test__()
                
            test_results["hardware_backend_status"] = "Success"
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["test_backend"] = error
            test_results["hardware_backend_status"] = "Failed"
            print(f"Error testing hardware backend: {str(e)}")
            print(traceback.format_exc())
        
        # Test IPFS accelerate endpoints
        try:
            print("Testing IPFS accelerate endpoints...")
            if self.ipfs_accelerate_py is None:
                raise ValueError("ipfs_accelerate_py is not initialized")
                
            results = {}
            
            # Initialize endpoints
            if not hasattr(self.ipfs_accelerate_py, "init_endpoints") or not callable(self.ipfs_accelerate_py.init_endpoints):
                raise AttributeError("ipfs_accelerate_py.init_endpoints method is not defined or not callable")
                
            print("Initializing endpoints...")
            # Get models list and validate it
            models_list = self.metadata.get('models', [])
            if not models_list:
                print("Warning: No models provided for init_endpoints")
                # Create an empty fallback structure
                ipfs_accelerate_init = {
                    "queues": {}, "queue": {}, "batch_sizes": {}, 
                    "endpoint_handler": {}, "consumer_tasks": {}, 
                    "caches": {}, "tokenizer": {},
                    "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                }
            else:
                # Try initialization with multi-tier fallback strategy
                try:
                    # First approach: Use properly structured resources with correct dictionary format
                    # Make sure resources are converted to proper dictionary structure
                    self._convert_resource_structures()
                    
                    # Use the structured_resources with correct nested format
                    print(f"Initializing endpoints for {len(models_list)} models using structured resources...")
                    ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(
                        models_list, 
                        self.resources.get("structured_resources", {})
                    )
                except Exception as e:
                    print(f"Error in first init_endpoints attempt with structured resources: {str(e)}")
                    try:
                        # Second approach: Use simplified endpoint structure
                        # Create a simple endpoint dictionary with correct structure
                        simple_endpoint = {
                            "endpoints": {
                                "local_endpoints": self.resources.get("local_endpoints", {}),
                                "libp2p_endpoints": self.resources.get("libp2p_endpoints", {}),
                                "tei_endpoints": self.resources.get("tei_endpoints", {})
                            },
                            "tokenizer": self.resources.get("tokenizer", {}),
                            "endpoint_handler": self.resources.get("endpoint_handler", {})
                        }
                        print(f"Trying second approach with simple_endpoint dictionary structure...")
                        ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, simple_endpoint)
                    except Exception as e2:
                        print(f"Error in second init_endpoints attempt: {str(e2)}")
                        try:
                            # Third approach: Create endpoint structure on-the-fly
                            # Do a fresh conversion with simplified structure
                            endpoint_resources = {}
                            
                            # Convert list-based resources to dict format where needed
                            for key, value in self.resources.items():
                                if isinstance(value, list) and key in ["local_endpoints", "tokenizer"]:
                                    # Convert list to dict for these specific resources
                                    if key == "local_endpoints":
                                        endpoints_dict = {}
                                        for entry in value:
                                            if len(entry) >= 2:
                                                model, endpoint_type = entry[0], entry[1]
                                                if model not in endpoints_dict:
                                                    endpoints_dict[model] = []
                                                endpoints_dict[model].append(entry)
                                        endpoint_resources[key] = endpoints_dict
                                    elif key == "tokenizer":
                                        tokenizers_dict = {}
                                        for entry in value:
                                            if len(entry) >= 2:
                                                model, endpoint_type = entry[0], entry[1]
                                                if model not in tokenizers_dict:
                                                    tokenizers_dict[model] = {}
                                                tokenizers_dict[model][endpoint_type] = None
                                        endpoint_resources[key] = tokenizers_dict
                                else:
                                    endpoint_resources[key] = value
                            
                            print(f"Trying third approach with on-the-fly conversion...")
                            ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, endpoint_resources)
                        except Exception as e3:
                            print(f"Error in third init_endpoints attempt: {str(e3)}")
                            # Final fallback - create a minimal viable endpoint structure for testing
                            print("Using fallback empty endpoint structure")
                            ipfs_accelerate_init = {
                                "queues": {}, "queue": {}, "batch_sizes": {}, 
                                "endpoint_handler": {}, "consumer_tasks": {}, 
                                "caches": {}, "tokenizer": {},
                                "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                            }
            
            # Test each model
            model_list = self.metadata.get('models', [])
            print(f"Testing {len(model_list)} models...")
            
            for model_idx, model in enumerate(model_list):
                print(f"Testing model {model_idx+1}/{len(model_list)}: {model}")
                
                if model not in results:
                    results[model] = {
                        "status": "Running",
                        "local_endpoint": {},
                        "api_endpoint": {}
                    }
                
                # Test local endpoint (tests both CUDA and OpenVINO internally)
                try:
                    print(f"  Testing local endpoint for {model}...")
                    local_result = await self.test_local_endpoint(model)
                    results[model]["local_endpoint"] = local_result
                    
                    # Determine if test was successful
                    if isinstance(local_result, dict) and not any("error" in str(k).lower() for k in local_result.keys()):
                        results[model]["local_endpoint_status"] = "Success"
                        
                        # Try to determine implementation type
                        impl_type = "MOCK"
                        for key, value in local_result.items():
                            if isinstance(value, dict) and "implementation_type" in value:
                                if "REAL" in value["implementation_type"]:
                                    impl_type = "REAL"
                                    break
                            elif isinstance(value, str) and "REAL" in value:
                                impl_type = "REAL"
                                break
                        
                        results[model]["local_endpoint_implementation"] = impl_type
                    else:
                        results[model]["local_endpoint_status"] = "Failed"
                except Exception as e:
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    results[model]["local_endpoint_error"] = error_info
                    results[model]["local_endpoint_status"] = "Failed"
                    print(f"  Error testing local endpoint for {model}: {str(e)}")
                
                # Test API endpoint
                try:
                    print(f"  Testing API endpoint for {model}...")
                    api_result = await self.test_api_endpoint(model)
                    results[model]["api_endpoint"] = api_result
                    
                    # Determine if test was successful
                    if isinstance(api_result, dict) and not any("error" in str(k).lower() for k in api_result.keys()):
                        results[model]["api_endpoint_status"] = "Success"
                        
                        # Try to determine implementation type
                        impl_type = "MOCK"
                        for key, value in api_result.items():
                            if isinstance(value, dict) and "implementation_type" in value:
                                if "REAL" in value["implementation_type"]:
                                    impl_type = "REAL"
                                    break
                            elif isinstance(value, str) and "REAL" in value:
                                impl_type = "REAL"
                                break
                        
                        results[model]["api_endpoint_implementation"] = impl_type
                    else:
                        results[model]["api_endpoint_status"] = "Failed"
                except Exception as e:
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    results[model]["api_endpoint_error"] = error_info
                    results[model]["api_endpoint_status"] = "Failed"
                    print(f"  Error testing API endpoint for {model}: {str(e)}")
                    
                # Determine overall model status
                if results[model].get("local_endpoint_status") == "Success" or results[model].get("api_endpoint_status") == "Success":
                    results[model]["status"] = "Success"
                else:
                    results[model]["status"] = "Failed"
            
            # Collect success/failure counts
            success_count = sum(1 for model_results in results.values() if model_results.get("status") == "Success")
            failure_count = sum(1 for model_results in results.values() if model_results.get("status") == "Failed")
            
            # Add summary data
            results["summary"] = {
                "total_models": len(model_list),
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": f"{success_count / len(model_list) * 100:.1f}%" if model_list else "N/A"
            }
            
            test_results["ipfs_accelerate_tests"] = results
            test_results["ipfs_accelerate_status"] = "Success"
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["ipfs_accelerate_tests"] = error
            test_results["ipfs_accelerate_status"] = "Failed"
            print(f"Error testing IPFS accelerate: {str(e)}")
            print(traceback.format_exc())

        # Set overall test status
        if (test_results.get("hardware_backend_status") == "Success" and 
            test_results.get("ipfs_accelerate_status") == "Success"):
            test_results["status"] = "Success"
        else:
            test_results["status"] = "Partial Success" if (test_results.get("hardware_backend_status") == "Success" or 
                                                           test_results.get("ipfs_accelerate_status") == "Success") else "Failed"

        return test_results

    async def test_local_endpoint(self, model, endpoint_list=None):
        """
        Test local endpoint for a model with proper error handling and resource cleanup.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if "local_endpoints" not in self.resources:
                return {"error": "Missing local_endpoints in resources"}
            if "tokenizer" not in self.resources:
                return {"error": "Missing tokenizer in resources"}
            
            # Convert resource structure if needed
            self._convert_resource_structures()
            
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "local_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "local_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("local_endpoints", {}):
                return {"error": f"Model {model} not found in local_endpoints"}
                
            # Get model endpoints from ipfs_accelerate_py
            local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["local_endpoints"][model]
            
            # Check if model exists in endpoint handler and tokenizer
            if not hasattr(self.ipfs_accelerate_py, "resources") or "endpoint_handler" not in self.ipfs_accelerate_py.resources:
                return {"error": "endpoint_handler not found in ipfs_accelerate_py.resources"}
                
            if model not in self.ipfs_accelerate_py.resources.get("endpoint_handler", {}):
                return {"error": f"Model {model} not found in endpoint_handler"}
                
            if "tokenizer" not in self.ipfs_accelerate_py.resources:
                return {"error": "tokenizer not found in ipfs_accelerate_py.resources"}
                
            if model not in self.ipfs_accelerate_py.resources.get("tokenizer", {}):
                return {"error": f"Model {model} not found in tokenizer"}
                
            # Get handlers and tokenizers for this model
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["endpoint_handler"][model]
            tokenizers_by_model = self.ipfs_accelerate_py.resources["tokenizer"][model]
            
            # Get available endpoint types
            endpoint_types = list(endpoint_handlers_by_model.keys())
            
            # Filter endpoints based on input or default behavior
            if endpoint_list is not None:
                # Filter by specified endpoint list
                local_endpoints_by_model_by_endpoint_list = [
                    x for x in local_endpoints_by_model 
                    if ("openvino:" in json.dumps(x) or "cuda:" in json.dumps(x)) 
                    and x[1] in endpoint_list 
                    and x[1] in endpoint_types
                ]
            else:
                # Use all CUDA and OpenVINO endpoints
                local_endpoints_by_model_by_endpoint_list = [
                    x for x in local_endpoints_by_model 
                    if ("openvino:" in json.dumps(x) or "cuda:" in json.dumps(x))
                    and x[1] in endpoint_types
                ]      
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint_list) == 0:
                return {"status": f"No valid endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint_list:
                # Clean up CUDA cache before testing to prevent memory issues
                if hasattr(self, "torch") and self.torch is not None and hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "empty_cache"):
                    self.torch.cuda.empty_cache()
                    print(f"  Cleared CUDA cache before testing endpoint {endpoint}")
                
                # Add timeout handling for model operations
                start_time = time.time()
                max_test_time = 300  # 5 minutes max per endpoint test
                
                # Get model type and validate it's supported
                try:
                    model_type = self.get_model_type(model)
                    if not model_type:
                        test_results[endpoint[1]] = {"error": f"Could not determine model type for {model}"}
                        continue
                        
                    # Load supported model types
                    hf_model_types_path = os.path.join(os.path.dirname(__file__), "hf_model_types.json")
                    if not os.path.exists(hf_model_types_path):
                        test_results[endpoint[1]] = {"error": "hf_model_types.json not found"}
                        continue
                        
                    with open(hf_model_types_path, "r") as f:
                        hf_model_types = json.load(f)
                        
                    method_name = "hf_" + model_type
                    
                    # Check if model type is supported
                    if model_type not in hf_model_types:
                        test_results[endpoint[1]] = {"error": f"Model type {model_type} not supported"}
                        continue
                        
                    # Check if endpoint exists in handlers
                    if endpoint[1] not in endpoint_handlers_by_model:
                        test_results[endpoint[1]] = {"error": f"Endpoint {endpoint[1]} not found for model {model}"}
                        continue
                        
                    endpoint_handler = endpoint_handlers_by_model[endpoint[1]]
                    
                    # Import the module and test the endpoint
                    try:
                        module = __import__('worker.skillset', fromlist=[method_name])
                        this_method = getattr(module, method_name)
                        this_hf = this_method(self.resources, self.metadata)
                        
                        # Check if test method is async and add timeout protection
                        if asyncio.iscoroutinefunction(this_hf.__test__):
                            try:
                                # Use asyncio.wait_for to add timeout protection
                                test = await asyncio.wait_for(
                                    this_hf.__test__(
                                        model, 
                                        endpoint_handlers_by_model[endpoint[1]], 
                                        endpoint[1], 
                                        tokenizers_by_model[endpoint[1]]
                                    ),
                                    timeout=max_test_time
                                )
                            except asyncio.TimeoutError:
                                test = {
                                    "error": f"Test timed out after {max_test_time} seconds",
                                    "timeout": True,
                                    "time_elapsed": time.time() - start_time
                                }
                        else:
                            # For sync methods, we still track time but don't have a clean timeout mechanism
                            # The global timeout in the Bash tool will still catch it if it runs too long
                            test = this_hf.__test__(
                                model, 
                                endpoint_handlers_by_model[endpoint[1]], 
                                endpoint[1], 
                                tokenizers_by_model[endpoint[1]]
                            )
                            # Check if we're past the timeout limit
                            if time.time() - start_time > max_test_time:
                                test = {
                                    "error": f"Test execution exceeded time limit of {max_test_time} seconds",
                                    "timeout": True,
                                    "time_elapsed": time.time() - start_time
                                }
                            
                        test_results[endpoint[1]] = test
                        
                        # Clean up resources
                        del this_hf
                        del this_method
                        del module
                        del test
                        
                        # Explicitly clean up CUDA memory
                        if hasattr(self, "torch") and self.torch is not None and hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "empty_cache"):
                            self.torch.cuda.empty_cache()
                            print(f"  Cleared CUDA cache after testing endpoint {endpoint[1]}")
                    except Exception as e:
                        test_results[endpoint[1]] = {
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
                except Exception as e:
                    test_results[endpoint[1]] = {
                        "error": f"Error processing endpoint {endpoint[1]}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_local_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_api_endpoint(self, model, endpoint_list=None):
        """
        Test API endpoints (TEI, OVMS) for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "tei_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing tei_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "tei_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "tei_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoint handlers"}
                
            local_endpoints = self.ipfs_accelerate_py.resources["tei_endpoints"]
            local_endpoints_types = [x[1] for x in local_endpoints]
            local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["tei_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["tei_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            local_endpoints_by_model_by_endpoint = [
                x for x in local_endpoints_by_model_by_endpoint 
                if x in local_endpoints_by_model 
                if x in local_endpoints_types
            ]
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid API endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # Try async call first, then fallback to sync
                    try:
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler("hello world")
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler("hello world")
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": test
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": test
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_api_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_libp2p_endpoint(self, model, endpoint_list=None):
        """
        Test libp2p endpoint for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "libp2p_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing libp2p_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "libp2p_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "libp2p_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("libp2p_endpoints", {}):
                return {"error": f"Model {model} not found in libp2p_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("libp2p_endpoints", {}):
                return {"error": f"Model {model} not found in libp2p_endpoint handlers"}
                
            libp2p_endpoints_by_model = self.ipfs_accelerate_py.endpoints["libp2p_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["libp2p_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid libp2p endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # Try async call first, then fallback to sync
                    try:
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler("hello world")
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler("hello world")
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": test
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": test
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_libp2p_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_qualcomm_endpoint(self, model, endpoint_list=None):
        """
        Test Qualcomm AI Engine endpoint for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        test_results = {}
        
        # Check if Qualcomm handler is available
        if "qualcomm_handler" not in dir(self):
            # Create handler if it doesn't exist
            self.qualcomm_handler = QualcommTestHandler()
            print(f"Created Qualcomm test handler (available: {self.qualcomm_handler.is_available()}, mock mode: {self.qualcomm_handler.mock_mode})")
        
        # If handler is still not available and we don't want to use mock mode, return error
        if not self.qualcomm_handler.is_available() and not os.environ.get("QUALCOMM_MOCK", "1") == "1":
            return {"error": "Qualcomm AI Engine not available and mock mode disabled"}
            
        # If the handler is not available, set mock mode
        if not self.qualcomm_handler.is_available():
            self.qualcomm_handler.mock_mode = True
            print("Using Qualcomm handler in mock mode for testing")
        
        try:
            # Get device information first
            device_info = self.qualcomm_handler.get_device_info()
            test_results["device_info"] = device_info
            
            # Determine model type from the model name with improved detection
            model_type = self._determine_model_type(model)
            test_results["model_type"] = model_type
            
            # Create appropriate sample input based on model type
            sample_input = self._create_sample_input(model_type)
            
            # Run inference with power monitoring and pass model type
            result = self.qualcomm_handler.run_inference(
                model, 
                sample_input, 
                monitor_metrics=True, 
                model_type=model_type
            )
            
            # Set status based on inference result
            if "error" in result:
                test_results["status"] = "Error"
                test_results["error"] = result["error"]
            else:
                test_results["status"] = "Success"
                test_results["implementation_type"] = f"QUALCOMM_{self.qualcomm_handler.sdk_type}"
                test_results["mock_mode"] = self.qualcomm_handler.mock_mode
                
                # Include inference output shape information
                if "output" in result and hasattr(result["output"], "shape"):
                    test_results["output_shape"] = str(result["output"].shape)
                
                # Add execution time if available
                if "execution_time_ms" in result:
                    test_results["execution_time_ms"] = result["execution_time_ms"]
                elif "metrics" in result and "execution_time_ms" in result["metrics"]:
                    test_results["execution_time_ms"] = result["metrics"]["execution_time_ms"]
                
                # Include throughput information if available
                if "throughput" in result:
                    test_results["throughput"] = result["throughput"]
                    if "throughput_units" in result:
                        test_results["throughput_units"] = result["throughput_units"]
                
                # Include metrics explicitly at the top level for better DB integration
                if "metrics" in result and isinstance(result["metrics"], dict):
                    # Store complete metrics
                    test_results["metrics"] = result["metrics"]
                    
                    # Also extract power metrics to a dedicated field for easier database storage
                    power_metrics = {}
                    
                    # Standard power fields
                    standard_fields = [
                        "power_consumption_mw", "energy_consumption_mj", "temperature_celsius", 
                        "monitoring_duration_ms", "average_power_mw", "peak_power_mw", "idle_power_mw"
                    ]
                    
                    # Enhanced metric fields from our updated implementation
                    enhanced_fields = [
                        "energy_efficiency_items_per_joule", "thermal_throttling_detected",
                        "battery_impact_percent_per_hour", "model_type"
                    ]
                    
                    # Combine all fields
                    all_fields = standard_fields + enhanced_fields
                    
                    # Extract fields that exist
                    for key in all_fields:
                        if key in result["metrics"]:
                            power_metrics[key] = result["metrics"][key]
                    
                    if power_metrics:
                        test_results["power_metrics"] = power_metrics
                        
                        # Add power efficiency summary information
                        if "energy_efficiency_items_per_joule" in power_metrics and "battery_impact_percent_per_hour" in power_metrics:
                            efficiency_summary = {
                                "energy_efficiency": power_metrics["energy_efficiency_items_per_joule"],
                                "battery_usage_per_hour": power_metrics["battery_impact_percent_per_hour"],
                                "power_consumption_mw": power_metrics.get("average_power_mw", power_metrics.get("power_consumption_mw")),
                                "thermal_management": "Throttling detected" if power_metrics.get("thermal_throttling_detected") else "Normal"
                            }
                            test_results["efficiency_summary"] = efficiency_summary
            
        except Exception as e:
            test_results["status"] = "Error"
            test_results["error"] = str(e)
            test_results["traceback"] = traceback.format_exc()
        
        return test_results
    
    def _determine_model_type(self, model_name):
        """Determine model type based on model name."""
        model_name = model_name.lower()
        
        if any(x in model_name for x in ["clip", "vit", "image", "resnet", "detr"]):
            return "vision"
        elif any(x in model_name for x in ["whisper", "wav2vec", "clap", "audio"]):
            return "audio"
        elif any(x in model_name for x in ["llava", "llama", "gpt", "llm"]):
            return "llm"
        else:
            return "text"  # Default to text embedding
    
    def _create_sample_input(self, model_type):
        """Create appropriate sample input based on model type."""
        import numpy as np
        
        if model_type == "vision":
            # Image tensor for vision models (batch_size, channels, height, width)
            return np.random.randn(1, 3, 224, 224).astype(np.float32)
        elif model_type == "audio":
            # Audio waveform for audio models (batch_size, samples)
            return np.random.randn(1, 16000).astype(np.float32)  # 1 second at 16kHz
        elif model_type == "llm":
            # Text prompt for language models
            return "This is a longer sample text for testing language models with the Qualcomm AI Engine. This text will be used for benchmarking inference performance on mobile hardware."
        else:
            # Simple text for embedding models
            return "This is a sample text for testing Qualcomm endpoint"
            
    async def test_ovms_endpoint(self, model, endpoint_list=None):
        """
        Test OpenVINO Model Server (OVMS) endpoints for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "ovms_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing ovms_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "ovms_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "ovms_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("ovms_endpoints", {}):
                return {"error": f"Model {model} not found in ovms_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("ovms_endpoints", {}):
                return {"error": f"Model {model} not found in ovms_endpoint handlers"}
                
            ovms_endpoints_by_model = self.ipfs_accelerate_py.endpoints["ovms_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["ovms_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid OVMS endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # Try async call first, then fallback to sync
                    # Since OVMS typically requires structured input, we'll create a simple tensor
                    try:
                        # Create a sample input (assuming a simple input tensor)
                        import numpy as np
                        sample_input = np.ones((1, 3, 224, 224), dtype=np.float32)  # Simple image-like tensor
                        
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler(sample_input)
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler(sample_input)
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": str(test)  # Convert numpy arrays to strings for JSON serialization
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                # Try with string input instead as a last resort
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": str(test)
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_ovms_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_webnn_endpoint(self, model, endpoint_list=None):
        """
        Test WebNN endpoint for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "webnn_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing webnn_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "webnn_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "webnn_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("webnn_endpoints", {}):
                return {"error": f"Model {model} not found in webnn_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("webnn_endpoints", {}):
                return {"error": f"Model {model} not found in webnn_endpoint handlers"}
                
            webnn_endpoints_by_model = self.ipfs_accelerate_py.endpoints["webnn_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["webnn_endpoints"][model]
            
            # Get list of valid endpoints for the model
            endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                endpoints_by_model_by_endpoint = [
                    x for x in endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid WebNN endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # Create appropriate input based on model type
                    model_type = self._determine_model_type(model)
                    sample_input = self._create_sample_input(model_type)
                    
                    # Try async call first, then fallback to sync
                    try:
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler(sample_input)
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler(sample_input)
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "model_type": model_type,
                            "result": str(test)[:100] + "..." if len(str(test)) > 100 else str(test)
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                # Try with string input instead as a last resort
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": str(test)[:100] + "..." if len(str(test)) > 100 else str(test)
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_webnn_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
        
    async def test_tei_endpoint(self, model, endpoint_list=None):
        """
        Test Text Embedding Inference (TEI) endpoints for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "tei_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing tei_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "tei_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "tei_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoint handlers"}
                
            local_endpoints = self.ipfs_accelerate_py.resources["tei_endpoints"]
            local_endpoints_types = [x[1] if isinstance(x, list) and len(x) > 1 else None for x in local_endpoints]
            local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["tei_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["tei_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid TEI endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # For TEI endpoints, we'll use a text sample
                    # Text Embedding Inference API typically expects a list of strings
                    try:
                        # Create a sample input
                        sample_input = ["This is a sample text for embedding generation."]
                        
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler(sample_input)
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler(sample_input)
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results, ensuring serializable format
                        if hasattr(test, "tolist"):  # Handle numpy arrays
                            result_data = test.tolist()
                        elif isinstance(test, list) and hasattr(test[0], "tolist"):  # List of numpy arrays
                            result_data = [item.tolist() if hasattr(item, "tolist") else item for item in test]
                        else:
                            result_data = test
                            
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": {
                                "shape": str(test.shape) if hasattr(test, "shape") else "unknown",
                                "type": str(type(test)),
                                "sample": str(result_data)[:100] + "..." if len(str(result_data)) > 100 else str(result_data)
                            }
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                # Try with single string input instead
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                # Format result for JSON serialization
                                if hasattr(test, "tolist"):  # Handle numpy arrays
                                    result_data = test.tolist()
                                elif isinstance(test, list) and hasattr(test[0], "tolist"):  # List of numpy arrays
                                    result_data = [item.tolist() if hasattr(item, "tolist") else item for item in test]
                                else:
                                    result_data = test
                                    
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": {
                                        "shape": str(test.shape) if hasattr(test, "shape") else "unknown",
                                        "type": str(type(test)),
                                        "sample": str(result_data)[:100] + "..." if len(str(result_data)) > 100 else str(result_data)
                                    }
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_tei_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_endpoint(self, model, endpoint=None):
        """
        Test a specific endpoint for a model.
        
        Args:
            model (str): The model to test
            endpoint (str, optional): The endpoint to test. Defaults to None.
            
        Returns:
            dict: Test results for the endpoint
        """
        test_results = {}
        
        try:
            # Test different endpoint types
            try:    
                test_results["local_endpoint"] = await self.test_local_endpoint(model, endpoint)
            except Exception as e:
                test_results["local_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["libp2p_endpoint"] = await self.test_libp2p_endpoint(model, endpoint)
            except Exception as e:
                test_results["libp2p_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["api_endpoint"] = await self.test_api_endpoint(model, endpoint)
            except Exception as e:
                test_results["api_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["ovms_endpoint"] = await self.test_ovms_endpoint(model, endpoint)
            except Exception as e:
                test_results["ovms_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["tei_endpoint"] = await self.test_tei_endpoint(model, endpoint)
            except Exception as e:
                test_results["tei_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            
            # Test Qualcomm endpoint if enabled
            if os.environ.get("TEST_QUALCOMM", "0") == "1":
                try:
                    test_results["qualcomm_endpoint"] = await self.test_qualcomm_endpoint(model, endpoint)
                except Exception as e:
                    test_results["qualcomm_endpoint"] = {
                        "status": "Error",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
            else:
                test_results["qualcomm_endpoint"] = {"status": "Not enabled", "info": "Set TEST_QUALCOMM=1 to enable"}
                
            # Test WebNN endpoint
            try:
                test_results["webnn_endpoint"] = await self.test_webnn_endpoint(model, endpoint)
            except Exception as e:
                test_results["webnn_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        except Exception as e:
            test_results["global_error"] = {
                "status": "Error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
        return test_results
        
    async def test_endpoints(self, models, endpoint_handler_object=None):
        """
        Test all available endpoints for each model.
        
        Args:
            models (list): List of models to test
            endpoint_handler_object (object, optional): Endpoint handler object. Defaults to None.
            
        Returns:
            dict: Test results for all endpoints
        """
        test_results = {}
        run_id = f"endpoint_test_{int(time.time())}"
        
        # Initialize DB handler for direct result storage
        db_handler = None
        if HAVE_DUCKDB and DEPRECATE_JSON_OUTPUT:
            try:
                db_path = os.environ.get("BENCHMARK_DB_PATH")
                db_handler = TestResultsDBHandler(db_path)
                if db_handler.is_available():
                    print(f"Database storage enabled - using run_id: {run_id}")
            except Exception as e:
                print(f"Error initializing database handler: {e}")
                db_handler = None
        
        # Track overall stats
        test_stats = {
            "total_models": len(models),
            "successful_tests": 0,
            "failed_tests": 0,
            "models_tested": []
        }
        
        # Test each model
        for model_idx, model in enumerate(models):
            print(f"Testing endpoints for model {model_idx+1}/{len(models)}: {model}")
            
            if model not in test_results:
                test_results[model] = {}
                
            model_success = True
            test_stats["models_tested"].append(model)
            
            # Test local endpoint (CUDA/OpenVINO)
            try: 
                print(f"  Testing local endpoint...")
                local_result = await self.test_local_endpoint(model)
                test_results[model]["local_endpoint"] = local_result
                
                # Store result directly in database if available
                if db_handler and isinstance(local_result, dict):
                    for endpoint_type, endpoint_results in local_result.items():
                        try:
                            db_handler.store_ipfs_acceleration_result(
                                model_name=model,
                                endpoint_type=endpoint_type,
                                acceleration_results=endpoint_results,
                                run_id=run_id
                            )
                            print(f"  Stored {endpoint_type} result directly in database")
                        except Exception as e:
                            print(f"  Error storing {endpoint_type} result in database: {e}")
                
                if isinstance(local_result, Exception) or (isinstance(local_result, dict) and any("Error" in str(v) for v in local_result.values())):
                    model_success = False
            except Exception as e:
                test_results[model]["local_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                model_success = False
                print(f"  Error testing local endpoint for {model}: {str(e)}")

            # Test WebNN endpoint (currently not implemented)
            try:
                test_results[model]["webnn_endpoint"] = {"status": "Not implemented"}
                # Store WebNN result in database
                if db_handler:
                    try:
                        db_handler.store_ipfs_acceleration_result(
                            model_name=model,
                            endpoint_type="webnn",
                            acceleration_results={"status": "Not implemented"},
                            run_id=run_id
                        )
                        print(f"  Stored WebNN result directly in database")
                    except Exception as e:
                        print(f"  Error storing WebNN result in database: {e}")
            except Exception as e:
                test_results[model]["webnn_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"  Error testing WebNN endpoint for {model}: {str(e)}")
                
            # Test WebGPU endpoint if enabled
            if os.environ.get("TEST_WEBGPU", "0") == "1":
                try:
                    print(f"  Testing WebGPU endpoint...")
                    # Placeholder for WebGPU implementation
                    test_results[model]["webgpu_endpoint"] = {"status": "Not implemented"}
                    # Store WebGPU result in database
                    if db_handler:
                        try:
                            db_handler.store_ipfs_acceleration_result(
                                model_name=model,
                                endpoint_type="webgpu",
                                acceleration_results={"status": "Not implemented"},
                                run_id=run_id
                            )
                            print(f"  Stored WebGPU result directly in database")
                        except Exception as e:
                            print(f"  Error storing WebGPU result in database: {e}")
                except Exception as e:
                    test_results[model]["webgpu_endpoint"] = {
                        "status": "Error",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    print(f"  Error testing WebGPU endpoint for {model}: {str(e)}")

            # Update test stats
            if model_success:
                test_stats["successful_tests"] += 1
            else:
                test_stats["failed_tests"] += 1

        # Add endpoint handler resources if provided
        if endpoint_handler_object:
            try:
                test_results["endpoint_handler_resources"] = endpoint_handler_object
            except Exception as e:
                test_results["endpoint_handler_resources"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
        # Add test stats to results
        test_results["test_stats"] = test_stats
        test_results["db_run_id"] = run_id if db_handler else None
                
        return test_results
    
    async def test_ipfs_accelerate(self):
        """
        Test IPFS accelerate endpoints for all models.
        
        Returns:
            dict: Test results for IPFS accelerate endpoints
        """
        test_results = {}
        
        try:
            print("Testing IPFS accelerate...")
            
            # Use the existing ipfs_accelerate_py instance
            if self.ipfs_accelerate_py is None:
                raise ValueError("ipfs_accelerate_py is not initialized")
                
            print("Initializing endpoints...")
            # Pass models explicitly when calling init_endpoints to avoid unbound 'model' error
            endpoint_resources = {}
            for key in self.resources:
                endpoint_resources[key] = self.resources[key]
                
            # Make resources a dict-like structure to avoid type issues
            if isinstance(endpoint_resources, list):
                endpoint_resources = {i: v for i, v in enumerate(endpoint_resources)}
                
            # Get models list and validate it
            models_list = self.metadata.get('models', [])
            if not models_list:
                print("Warning: No models provided for init_endpoints")
                # Create an empty fallback structure
                ipfs_accelerate_init = {
                    "queues": {}, "queue": {}, "batch_sizes": {}, 
                    "endpoint_handler": {}, "consumer_tasks": {}, 
                    "caches": {}, "tokenizer": {},
                    "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                }
            else:
                # Try the initialization with different approaches
                try:
                    print(f"Initializing endpoints for {len(models_list)} models...")
                    ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, endpoint_resources)
                except Exception as e:
                    print(f"Error in first init_endpoints attempt: {str(e)}")
                    try:
                        # Alternative approach - creating a simple endpoint structure with actual resource data
                        simple_endpoint = {
                            "local_endpoints": self.resources.get("local_endpoints", []),
                            "libp2p_endpoints": self.resources.get("libp2p_endpoints", []),
                            "tei_endpoints": self.resources.get("tei_endpoints", [])
                        }
                        print(f"Trying second approach with simple_endpoint structure")
                        ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, simple_endpoint)
                    except Exception as e2:
                        print(f"Error in second init_endpoints attempt: {str(e2)}")
                        # Final fallback - create a minimal viable endpoint structure
                        print("Using fallback empty endpoint structure")
                        ipfs_accelerate_init = {
                            "queues": {}, "queue": {}, "batch_sizes": {}, 
                            "endpoint_handler": {}, "consumer_tasks": {}, 
                            "caches": {}, "tokenizer": {},
                            "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                        }
            
            # Test endpoints for all models
            model_list = self.metadata.get('models', [])
            print(f"Testing endpoints for {len(model_list)} models...")
            
            test_endpoints = await self.test_endpoints(model_list, ipfs_accelerate_init)
            test_results["test_endpoints"] = test_endpoints
            test_results["status"] = "Success"
            
            return test_results
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["error"] = error
            test_results["status"] = "Failed"
            print(f"Error testing IPFS accelerate endpoints: {str(e)}")
            print(traceback.format_exc())
            
            return test_results
    
    async def __test__(self, resources=None, metadata=None):
        """
        Main test entry point that runs all tests and collects results.
        
        This method follows the 4-phase testing approach defined in the class documentation:
        - Phase 1: Test with models defined in global metadata
        - Phase 2: Test with models from mapped_models.json
        - Phase 3: Collect and analyze test results
        - Phase 4: Generate test reports
        
        Args:
            resources (dict, optional): Dictionary of resources. Defaults to None.
            metadata (dict, optional): Dictionary of metadata. Defaults to None.
            
        Returns:
            dict: Comprehensive test results
        """
        start_time = time.time()
        start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting test suite at {start_time_str}")
        
        # Initialize Qualcomm test handler if needed
        test_qualcomm = os.environ.get("TEST_QUALCOMM", "0") == "1"
        if test_qualcomm:
            if "qualcomm_handler" not in dir(self):
                self.qualcomm_handler = QualcommTestHandler()
                if self.qualcomm_handler.is_available():
                    print(f"Qualcomm AI Engine detected: {self.qualcomm_handler.sdk_type} SDK")
                    # Add to resources if available
                    if "qualcomm" not in self.resources:
                        self.resources["qualcomm"] = self.qualcomm_handler
                    
                    # Add qualcomm_handler as an endpoint type
                    for model in self.metadata.get("models", []):
                        if "local_endpoints" in self.resources and model in self.resources["local_endpoints"]:
                            # Add qualcomm endpoint to local_endpoints
                            qualcomm_endpoint = [model, "qualcomm:0", 32768]
                            if qualcomm_endpoint not in self.resources["local_endpoints"][model]:
                                self.resources["local_endpoints"][model].append(qualcomm_endpoint)
                                print(f"Added Qualcomm endpoint for model {model}")
                                
                            # Add tokenizer entry for qualcomm endpoint
                            if "tokenizer" in self.resources and model in self.resources["tokenizer"]:
                                self.resources["tokenizer"][model]["qualcomm:0"] = None
                                
                            # Add endpoint handler entry for qualcomm endpoint
                            if "endpoint_handler" in self.resources and model in self.resources["endpoint_handler"]:
                                self.resources["endpoint_handler"][model]["qualcomm:0"] = None
        
        # Initialize resources if not provided
        if resources is not None:
            self.resources = resources
        if metadata is not None:
            self.metadata = metadata
            
        # Ensure required resource dictionaries exist
        required_resource_keys = [
            "local_endpoints", "tei_endpoints", "libp2p_endpoints", 
            "openvino_endpoints", "tokenizer", "endpoint_handler"
        ]
        
        print("Initializing resources...")
        for key in required_resource_keys:
            if key not in self.resources:
                self.resources[key] = {}
                print(f"  Created empty {key} dictionary")
            
        # Load mapped models from JSON
        mapped_models = {}
        mapped_models_values = []
        mapped_models_path = os.path.join(os.path.dirname(__file__), "mapped_models.json")
        
        print("Loading mapped models...")
        if os.path.exists(mapped_models_path):
            try:
                with open(mapped_models_path, "r") as f:
                    mapped_models = json.load(f)
                mapped_models_values = list(mapped_models.values())
                print(f"  Loaded {len(mapped_models)} model mappings")
                
                # Update metadata with models from mapped_models.json
                if "models" not in self.metadata or not self.metadata["models"]:
                    self.metadata["models"] = mapped_models_values
                    print("  Updated self.metadata with mapped models")
            except Exception as e:
                print(f"Error loading mapped_models.json: {str(e)}")
                print(traceback.format_exc())
        else:
            print("  Warning: mapped_models.json not found")
        
        # Initialize the transformers module if available
        if "transformers" not in self.resources:
            try:
                import transformers
                self.resources["transformers"] = transformers
                print("  Added transformers module to resources")
            except ImportError:
                from unittest.mock import MagicMock
                self.resources["transformers"] = MagicMock()
                print("  Added mock transformers module to resources")
        
        # Setup endpoints for each model and hardware platform
        endpoint_types = ["cuda:0", "openvino:0", "cpu:0"]
        endpoint_count = 0
        
        # Handle both list and dictionary resource structures
        if isinstance(self.resources["local_endpoints"], list):
            # Convert list to empty dictionary to prepare for structured data
            self.resources["local_endpoints"] = {}
            self.resources["tokenizer"] = {}
            
        # Add required resource dictionaries for init_endpoints
        required_queue_keys = ["queue", "queues", "batch_sizes", "consumer_tasks", "caches"]
        for key in required_queue_keys:
            if key not in self.resources:
                self.resources[key] = {}
            
        print("Setting up endpoints for each model...")
        if "models" in self.metadata and self.metadata["models"]:
            if "endpoints" not in self.resources:
                self.resources["endpoints"] = {}
                
            if "local_endpoints" not in self.resources["endpoints"]:
                self.resources["endpoints"]["local_endpoints"] = {}
                
            # Make sure we have a direct local_endpoints reference for backward compatibility
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or not self.ipfs_accelerate_py.endpoints:
                self.ipfs_accelerate_py.endpoints = {
                    "local_endpoints": {},
                    "api_endpoints": {},
                    "libp2p_endpoints": {}
                }
            
            for model in self.metadata["models"]:
                # Create model entry in endpoints dictionary
                if model not in self.resources["local_endpoints"]:
                    self.resources["local_endpoints"][model] = []
                
                # Create model entry inside endpoints structure
                if model not in self.resources["endpoints"]["local_endpoints"]:
                    self.resources["endpoints"]["local_endpoints"][model] = []
                
                # Also initialize in the ipfs_accelerate_py.endpoints structure
                if model not in self.ipfs_accelerate_py.endpoints["local_endpoints"]:
                    self.ipfs_accelerate_py.endpoints["local_endpoints"][model] = []
                
                # Create model entry in tokenizer and endpoint_handler dictionaries
                if model not in self.resources["tokenizer"]:
                    self.resources["tokenizer"][model] = {}
                    
                if model not in self.resources["endpoint_handler"]:
                    self.resources["endpoint_handler"][model] = {}
                
                # Make sure ipfs_accelerate_py has the same entries in its resources
                if not hasattr(self.ipfs_accelerate_py, "resources"):
                    self.ipfs_accelerate_py.resources = {}
                
                for resource_key in ["tokenizer", "endpoint_handler", "queue", "queues", "batch_sizes"]:
                    if resource_key not in self.ipfs_accelerate_py.resources:
                        self.ipfs_accelerate_py.resources[resource_key] = {}
                    
                    if model not in self.ipfs_accelerate_py.resources[resource_key]:
                        self.ipfs_accelerate_py.resources[resource_key][model] = {} if resource_key != "queue" else asyncio.Queue(128)
                
                for endpoint in endpoint_types:
                    # Create endpoint info (model, endpoint, context_length)
                    endpoint_info = [model, endpoint, 32768]
                    
                    # Avoid duplicate entries in resources["local_endpoints"]
                    if endpoint_info not in self.resources["local_endpoints"][model]:
                        self.resources["local_endpoints"][model].append(endpoint_info)
                    
                    # Also add to resources["endpoints"]["local_endpoints"]
                    if endpoint_info not in self.resources["endpoints"]["local_endpoints"][model]:
                        self.resources["endpoints"]["local_endpoints"][model].append(endpoint_info)
                    
                    # Add to ipfs_accelerate_py.endpoints["local_endpoints"]
                    if endpoint_info not in self.ipfs_accelerate_py.endpoints["local_endpoints"][model]:
                        self.ipfs_accelerate_py.endpoints["local_endpoints"][model].append(endpoint_info)
                    
                    # Add tokenizer entry for this model-endpoint combination
                    if endpoint not in self.resources["tokenizer"][model]:
                        self.resources["tokenizer"][model][endpoint] = None
                    
                    # Add tokenizer to ipfs_accelerate_py.resources
                    if endpoint not in self.ipfs_accelerate_py.resources["tokenizer"][model]:
                        self.ipfs_accelerate_py.resources["tokenizer"][model][endpoint] = None
                    
                    # Add endpoint handler entry
                    if endpoint not in self.resources["endpoint_handler"][model]:
                        self.resources["endpoint_handler"][model][endpoint] = None
                    
                    # Add endpoint handler to ipfs_accelerate_py.resources
                    if endpoint not in self.ipfs_accelerate_py.resources["endpoint_handler"][model]:
                        self.ipfs_accelerate_py.resources["endpoint_handler"][model][endpoint] = None
                        
                        # Create a mock handler directly in ipfs_accelerate_py
                        if hasattr(self.ipfs_accelerate_py, "_create_mock_handler"):
                            try:
                                self.ipfs_accelerate_py._create_mock_handler(model, endpoint)
                                print(f"  Created mock handler for {model} with {endpoint}")
                            except Exception as e:
                                print(f"  Error creating mock handler: {str(e)}")
                    
                    # Track for reporting
                    endpoint_count += 1
            
            print(f"  Added {endpoint_count} endpoints for {len(self.metadata['models'])} models")
            
            # Create properly structured resources dictionary with all required keys
            self.resources["structured_resources"] = {
                "tokenizer": self.resources["tokenizer"],
                "endpoint_handler": self.resources["endpoint_handler"],
                "queue": self.resources.get("queue", {}),  # Add queue dictionary - required by init_endpoints
                "queues": self.resources.get("queues", {}), # Add queues dictionary
                "batch_sizes": self.resources.get("batch_sizes", {}), # Add batch_sizes dictionary
                "consumer_tasks": self.resources.get("consumer_tasks", {}), # Add consumer_tasks dictionary
                "caches": self.resources.get("caches", {}), # Add caches dictionary
                "endpoints": {
                    "local_endpoints": self.resources["local_endpoints"],
                    "api_endpoints": self.resources.get("tei_endpoints", {}),
                    "libp2p_endpoints": self.resources.get("libp2p_endpoints", {})
                }
            }
            
            # Debugging: Print structure information
            print(f"  Endpoints structure: Dictionary with {len(self.resources['local_endpoints'])} models")
            print(f"  Tokenizer structure: Dictionary with {len(self.resources['tokenizer'])} models")
            print(f"  Endpoint handler structure: Dictionary with {len(self.resources['endpoint_handler'])} models")
        else:
            print("  Warning: No models found in metadata")
        
        # Prepare test results structure
        test_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_date": start_time,
                "status": "Running",
                "test_phases_completed": 0,
                "total_test_phases": 4 if mapped_models else 2
            },
            "models_tested": {
                "global_models": len(self.metadata.get("models", [])),
                "mapped_models": len(mapped_models)
            },
            "configuration": {
                "endpoint_types": endpoint_types,
                "model_count": len(self.metadata.get("models", [])),
                "endpoints_per_model": len(endpoint_types)
            }
        }
        
        # Run the tests in phases
        try:
            # Phase 1: Test with models in global metadata
            print("\n=== PHASE 1: Testing with global metadata models ===")
            if not self.metadata.get("models"):
                print("No models in global metadata, skipping Phase 1")
                test_results["phase1_global_models"] = {"status": "Skipped", "reason": "No models in global metadata"}
            else:
                # Limit to first 2 models to avoid timeouts
                original_models = self.metadata.get('models', []).copy()
                self.metadata["models"] = self.metadata.get("models", [])[:2]
                print(f"Testing {len(self.metadata.get('models', []))} models from global metadata (limited to first 2 for speed)")
                test_results["phase1_global_models"] = await self.test()
                self.metadata["models"] = original_models  # Restore the original list
                test_results["metadata"]["test_phases_completed"] += 1
                print(f"Phase 1 completed with status: {test_results['phase1_global_models'].get('status', 'Unknown')}")
            
            # Phase 2: Test with mapped models from JSON file
            print("\n=== PHASE 2: Testing with mapped models ===")
            if not mapped_models:
                print("No mapped models found, skipping Phase 2")
                test_results["phase2_mapped_models"] = {"status": "Skipped", "reason": "No mapped models found"}
            else:
                # Save original models list
                original_models = self.metadata.get("models", [])
                
                # Update metadata to use a limited subset of mapped models (first 2)
                limited_models = mapped_models_values[:2]
                print(f"Testing {len(limited_models)} models from mapped_models.json (limited to first 2 for speed)")
                self.metadata["models"] = limited_models
                test_results["phase2_mapped_models"] = await self.test()
                
                # Restore original models list
                self.metadata["models"] = original_models
                test_results["metadata"]["test_phases_completed"] += 1
                print(f"Phase 2 completed with status: {test_results['phase2_mapped_models'].get('status', 'Unknown')}")
            
            # Phase 3: Analyze test results
            print("\n=== PHASE 3: Analyzing test results ===")
            analysis = {
                "model_coverage": {},
                "platform_performance": {
                    "cuda": {"success": 0, "failure": 0, "success_rate": "0%"},
                    "openvino": {"success": 0, "failure": 0, "success_rate": "0%"}
                },
                "implementation_types": {
                    "REAL": 0,
                    "MOCK": 0,
                    "Unknown": 0
                }
            }
            
            # Analyze Phase 1 results
            if "phase1_global_models" in test_results and "ipfs_accelerate_tests" in test_results["phase1_global_models"]:
                phase1_results = test_results["phase1_global_models"]["ipfs_accelerate_tests"]
                if isinstance(phase1_results, dict) and "summary" in phase1_results:
                    # Process model results
                    for model, model_results in phase1_results.items():
                        if model == "summary":
                            continue
                            
                        # Track model success/failure
                        if model not in analysis["model_coverage"]:
                            analysis["model_coverage"][model] = {"status": model_results.get("status", "Unknown")}
                            
                        # Track platform performance
                        for platform in ["cuda", "openvino"]:
                            if platform in model_results and "status" in model_results[platform]:
                                if model_results[platform]["status"] == "Success":
                                    analysis["platform_performance"][platform]["success"] += 1
                                else:
                                    analysis["platform_performance"][platform]["failure"] += 1
                                    
                        # Track implementation types
                        for platform in ["cuda", "openvino"]:
                            if platform in model_results and "implementation_type" in model_results[platform]:
                                impl_type = model_results[platform]["implementation_type"]
                                if "REAL" in impl_type:
                                    analysis["implementation_types"]["REAL"] += 1
                                elif "MOCK" in impl_type:
                                    analysis["implementation_types"]["MOCK"] += 1
                                else:
                                    analysis["implementation_types"]["Unknown"] += 1
            
            # Calculate success rates
            for platform in ["cuda", "openvino"]:
                platform_data = analysis["platform_performance"][platform]
                total = platform_data["success"] + platform_data["failure"]
                if total > 0:
                    platform_data["success_rate"] = f"{(platform_data['success'] / total) * 100:.1f}%"
            
            # Add analysis to test results
            test_results["phase3_analysis"] = analysis
            test_results["metadata"]["test_phases_completed"] += 1
            print("Analysis completed")
            
            # Phase 4: Generate test report
            print("\n=== PHASE 4: Generating test report ===")
            
            # Create test report summary
            report = {
                "summary": {
                    "test_date": start_time,
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "models_tested": test_results["models_tested"],
                    "phases_completed": test_results["metadata"]["test_phases_completed"],
                    "platform_performance": analysis["platform_performance"],
                    "implementation_breakdown": analysis["implementation_types"]
                },
                "recommendations": []
            }
            
            # Add recommendations based on analysis
            if analysis["implementation_types"]["MOCK"] > analysis["implementation_types"]["REAL"]:
                report["recommendations"].append("Focus on implementing more REAL implementations to replace MOCK implementations")
                
            if analysis["platform_performance"]["cuda"]["success_rate"] < "50%":
                report["recommendations"].append("Improve CUDA platform support for better performance")
                
            if analysis["platform_performance"]["openvino"]["success_rate"] < "50%":
                report["recommendations"].append("Improve OpenVINO platform support for better compatibility")
            
            # Add report to test results
            test_results["phase4_report"] = report
            test_results["metadata"]["test_phases_completed"] += 1
            print("Test report generated")
            
            # Update overall test status
            if (test_results["metadata"]["test_phases_completed"] == 
                test_results["metadata"]["total_test_phases"]):
                test_results["metadata"]["status"] = "Success"
            else:
                test_results["metadata"]["status"] = "Partial Success"
                
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["error"] = error
            test_results["metadata"]["status"] = "Failed"
            print(f"Error running tests: {str(e)}")
            print(traceback.format_exc())
        
        # Record execution time
        execution_time = time.time() - start_time
        if "metadata" in test_results:
            test_results["metadata"]["execution_time"] = execution_time
            
        # Add Qualcomm metrics if available
        if "qualcomm_handler" in dir(self) and self.qualcomm_handler.is_available():
            device_info = self.qualcomm_handler.get_device_info()
            if "metadata" not in test_results:
                test_results["metadata"] = {}
            test_results["metadata"]["qualcomm_device_info"] = device_info
        
        # Save test results to database and file
        print("\nSaving test results...")
        this_file = os.path.abspath(sys.modules[__name__].__file__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Always try to store in database first (DuckDB is preferred storage method)
        db_result_saved = False
        if HAVE_DUCKDB and not os.environ.get("DISABLE_DB_STORAGE", "0") == "1":
            try:
                # Initialize database handler
                db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH")
                db_handler = TestResultsDBHandler(db_path)
                
                if db_handler.is_available():
                    # Generate run ID
                    run_id = f"test_run_{timestamp}"
                    
                    # Store results
                    success = db_handler.store_test_results(test_results, run_id)
                    if success:
                        print(f"Saved test results to database with run ID: {run_id}")
                        
                        # Store IPFS acceleration results specifically
                        print("Processing IPFS acceleration results...")
                        db_handler._store_ipfs_acceleration_results(test_results, run_id)
                        
                        # Process test endpoint results
                        for model_name, model_data in test_results.get("test_endpoints", {}).items():
                            # Skip non-model entries like test_stats
                            if model_name in ["test_stats", "endpoint_handler_resources"]:
                                continue
                                
                            # Process local endpoint results
                            if "local_endpoint" in model_data and isinstance(model_data["local_endpoint"], dict):
                                for endpoint_type, endpoint_results in model_data["local_endpoint"].items():
                                    db_handler.store_ipfs_acceleration_result(
                                        model_name=model_name,
                                        endpoint_type=endpoint_type,
                                        acceleration_results=endpoint_results,
                                        run_id=run_id
                                    )
                            
                            # Process other endpoint types
                            for endpoint_key in ["qualcomm_endpoint", "webnn_endpoint", "webgpu_endpoint"]:
                                if endpoint_key in model_data and isinstance(model_data[endpoint_key], dict):
                                    endpoint_type = endpoint_key.replace("_endpoint", "")
                                    db_handler.store_ipfs_acceleration_result(
                                        model_name=model_name,
                                        endpoint_type=endpoint_type,
                                        acceleration_results=model_data[endpoint_key],
                                        run_id=run_id
                                    )
                        
                        db_result_saved = True
                        
                        # Generate a report immediately after storing results
                        try:
                            report_path = "test_report.md"
                            report_result = db_handler.generate_report(format="markdown", output_file=report_path)
                            if report_result:
                                print(f"Test report generated: {report_path}")
                                
                            # Generate an IPFS acceleration report
                            accel_report_path = "ipfs_acceleration_report.html"
                            accel_report = db_handler.generate_ipfs_acceleration_report(
                                format="html", 
                                output=accel_report_path,
                                run_id=run_id
                            )
                            if accel_report:
                                print(f"IPFS acceleration report generated: {accel_report_path}")
                        except Exception as e:
                            print(f"Error generating automatic reports: {e}")
                    else:
                        print("Failed to save test results to database")
                else:
                    print("Database storage not available, falling back to JSON")
            except Exception as e:
                print(f"Error saving test results to database: {e}")
                print(traceback.format_exc())
        
        # Check if we should skip JSON output (db-only mode or successful DB storage with deprecated JSON)
        skip_json = args.db_only or (DEPRECATE_JSON_OUTPUT and db_result_saved)
        
        # Save to JSON file if not skipping JSON output
        if not skip_json:
            test_log = os.path.join(os.path.dirname(this_file), f"test_results_{timestamp}.json")
            try:
                with open(test_log, "w") as f:
                    json.dump(test_results, f, indent=4)
                print(f"Saved detailed test results to {test_log}")
                    
                # Also save to standard test_results.json for backward compatibility
                standard_log = os.path.join(os.path.dirname(this_file), "test_results.json")
                with open(standard_log, "w") as f:
                    json.dump(test_results, f, indent=4)
                print(f"Saved test results to {standard_log}")
            except Exception as e:
                print(f"Error saving test results: {str(e)}")
        else:
            if args.db_only:
                print("JSON output disabled by --db-only flag - results stored in database only")
            else:
                print("JSON output deprecated in favor of database storage - results stored in database only")
        
        print(f"\nTest suite completed with status: {test_results['metadata']['status']}")
        return test_results


if __name__ == "__main__":
    """
    Main entry point for the test_ipfs_accelerate script.
    
    This will initialize the test class with a list of models to test,
    setup the necessary resources, and run the test suite.
    
    Command line arguments:
    --report: Generate a general report from the database (formats: markdown, html, json)
    --ipfs-acceleration-report: Generate an IPFS acceleration-specific report
    --comparison-report: Generate a comparative report for acceleration types across models
    --webgpu-analysis: Generate detailed WebGPU performance analysis report
    --output: Path to save the report (default: <report_type>.<format>)
    --format: Report format (default: markdown, html recommended for comparison reports)
    --db-path: Path to the database (default: from environment or ./benchmark_db.duckdb)
    --run-id: Specific run ID to generate a report for (default: latest)
    --model: Specific model name to filter results for comparison report
    --models: Comma-separated list of models to test (default: 2 small embedding models)
    --endpoints: Comma-separated list of endpoint types to test
    --qualcomm: Include Qualcomm endpoints in testing
    --webnn: Include WebNN endpoints in testing
    --webgpu: Include WebGPU endpoints in testing
    --browser: Specify browser for WebGPU/WebNN analysis (chrome, firefox, edge, safari)
    --shader-metrics: Include shader compilation metrics in WebGPU analysis
    --compute-shader-optimization: Analyze compute shader optimizations for WebGPU
    --store-in-db: Store test results directly in database even if DEPRECATE_JSON_OUTPUT=0
    --db-only: Only store results in database, never in JSON (overrides DEPRECATE_JSON_OUTPUT=0)
    
    Examples:
      # Run tests with default models
      python test_ipfs_accelerate.py
      
      # Run tests with specific models, including WebNN, WebGPU and Qualcomm endpoints
      python test_ipfs_accelerate.py --models "bert-base-uncased,prajjwal1/bert-tiny" --webnn --webgpu --qualcomm
      
      # Run tests and store results only in the database (no JSON files)
      python test_ipfs_accelerate.py --models "bert-base-uncased" --db-only
      
      # Run tests with custom database path
      python test_ipfs_accelerate.py --db-path ./my_benchmark.duckdb
      
      # Generate IPFS acceleration report in HTML format
      python test_ipfs_accelerate.py --ipfs-acceleration-report --format html --output accel_report.html
      
      # Generate comparison report for all models
      python test_ipfs_accelerate.py --comparison-report --format html
      
      # Generate comparison report for a specific model
      python test_ipfs_accelerate.py --comparison-report --model "bert-base-uncased"
      
      # Generate WebGPU analysis report
      python test_ipfs_accelerate.py --webgpu-analysis --browser firefox --shader-metrics
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test IPFS Accelerate Python")
    parser.add_argument("--report", action="store_true", help="Generate a general report from the database")
    parser.add_argument("--ipfs-acceleration-report", action="store_true", help="Generate IPFS acceleration specific report")
    parser.add_argument("--comparison-report", action="store_true", help="Generate acceleration comparison report across models or for a specific model")
    parser.add_argument("--output", help="Path to save the report")
    parser.add_argument("--format", choices=["markdown", "html", "json"], default="markdown", help="Report format")
    parser.add_argument("--db-path", help="Path to the database")
    parser.add_argument("--run-id", help="Specific run ID to generate a report for")
    parser.add_argument("--model", help="Specific model name to filter results for comparison report")
    parser.add_argument("--models", help="Comma-separated list of models to test")
    parser.add_argument("--endpoints", help="Comma-separated list of endpoint types to test")
    parser.add_argument("--qualcomm", action="store_true", help="Include Qualcomm endpoints in testing")
    parser.add_argument("--webnn", action="store_true", help="Include WebNN endpoints in testing")
    parser.add_argument("--webgpu", action="store_true", help="Include WebGPU endpoints in testing")
    parser.add_argument("--webgpu-analysis", action="store_true", help="Generate detailed WebGPU performance analysis report")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge", "safari"], help="Specify browser for WebGPU/WebNN analysis")
    parser.add_argument("--shader-metrics", action="store_true", help="Include shader compilation metrics in WebGPU analysis")
    parser.add_argument("--compute-shader-optimization", action="store_true", help="Analyze compute shader optimizations for WebGPU")
    parser.add_argument("--store-in-db", action="store_true", help="Store test results directly in database even if DEPRECATE_JSON_OUTPUT=0")
    parser.add_argument("--db-only", action="store_true", help="Only store results in database, never in JSON (overrides DEPRECATE_JSON_OUTPUT=0)")
    args = parser.parse_args()
    
    # If generating a report, use the database handler directly
    if args.report or args.ipfs_acceleration_report or args.comparison_report:
        # Initialize database handler
        db_handler = TestResultsDBHandler(db_path=args.db_path)
        
        if db_handler.con is None:
            print("Error: Database not available. Install DuckDB or check database path.")
            sys.exit(1)
        
        # Determine output path if not provided
        if not args.output:
            if args.ipfs_acceleration_report:
                args.output = f"ipfs_acceleration_report.{args.format}"
            elif args.comparison_report:
                args.output = f"acceleration_comparison_report.{args.format}"
            else:
                args.output = f"test_report.{args.format}"
        
        # Generate specific type of report based on arguments
        if args.ipfs_acceleration_report:
            report_result = db_handler.generate_ipfs_acceleration_report(
                format=args.format, 
                output=args.output,
                run_id=args.run_id
            )
            report_type = "IPFS acceleration"
        elif args.comparison_report:
            # For comparison report, HTML is most useful due to visualizations
            if args.format != "html" and args.format != "json":
                print("Warning: Comparison report works best with HTML format. Switching to HTML.")
                args.format = "html"
                if args.output.endswith(".md") or args.output.endswith(".markdown"):
                    args.output = args.output.rsplit(".", 1)[0] + ".html"
            
            report_result = db_handler.generate_acceleration_comparison_report(
                format=args.format,
                output=args.output,
                model_name=args.model
            )
            report_type = "Acceleration comparison"
        elif args.webgpu_analysis:
            # For WebGPU analysis report, HTML is most useful for visualization
            if args.format != "html" and args.format != "json":
                print("Warning: WebGPU analysis report works best with HTML format. Switching to HTML.")
                args.format = "html"
                if args.output and (args.output.endswith(".md") or args.output.endswith(".markdown")):
                    args.output = args.output.rsplit(".", 1)[0] + ".html"
            
            # Default output name if not provided
            if not args.output:
                browser_suffix = f"_{args.browser}" if args.browser else ""
                args.output = f"webgpu_analysis{browser_suffix}.{args.format}"
            
            report_result = db_handler.generate_webgpu_analysis_report(
                format=args.format,
                output=args.output,
                browser=args.browser,
                include_shader_metrics=args.shader_metrics,
                analyze_compute_shaders=args.compute_shader_optimization
            )
            report_type = "WebGPU analysis"
        else:
            report_result = db_handler.generate_report(
                format=args.format, 
                output_file=args.output
            )
            report_type = "general"
        
        if not report_result:
            print(f"Error generating {report_type} report.")
            sys.exit(1)
            
        print(f"{report_type.capitalize()} report successfully generated and saved to {args.output}")
        sys.exit(0)
    
    # Define metadata including models to test
    # For quick testing, we'll use just 2 small models by default
    default_models = [
        "BAAI/bge-small-en-v1.5",
        "prajjwal1/bert-tiny"
    ]
    
    # Use models from command line if provided
    if args.models:
        test_models = args.models.split(",")
    else:
        test_models = default_models
    
    # Use endpoints from command line if provided
    if args.endpoints:
        endpoint_types = args.endpoints.split(",")
    else:
        endpoint_types = ["cuda:0", "openvino:0", "cpu:0"]
        
    # Add Qualcomm endpoint if requested
    if args.qualcomm:
        endpoint_types.append("qualcomm:0")
        os.environ["TEST_QUALCOMM"] = "1"
        
    # Add WebNN endpoint if requested
    if args.webnn:
        endpoint_types.append("webnn:0")
        os.environ["TEST_WEBNN"] = "1"
        
    # Add WebGPU endpoint if requested
    if args.webgpu:
        endpoint_types.append("webgpu:0")
        os.environ["TEST_WEBGPU"] = "1"
    
    metadata = {
        "dataset": "laion/gpt4v-dataset",
        "namespace": "laion/gpt4v-dataset",
        "column": "link",
        "role": "master",
        "split": "train",
        "models": test_models,
        "chunk_settings": {},
        "path": "/storage/gpt4v-dataset/data",
        "dst_path": "/storage/gpt4v-dataset/data",
    }
    
    # Initialize resources with proper dictionary structures
    resources = {
        "local_endpoints": {},
        "tei_endpoints": {},
        "tokenizer": {},
        "endpoint_handler": {}
    }
    
    # Define endpoint types and initialize with dictionary structure
    for model in metadata["models"]:
        # Initialize model dictionaries
        resources["local_endpoints"][model] = []
        resources["tokenizer"][model] = {}
        resources["endpoint_handler"][model] = {}
        
        # Add endpoints for each model and endpoint type
        for endpoint in endpoint_types:
            # Add endpoint entry
            resources["local_endpoints"][model].append([model, endpoint, 32768])
            
            # Initialize tokenizer and endpoint handler entries
            resources["tokenizer"][model][endpoint] = None
            resources["endpoint_handler"][model][endpoint] = None
    
    # Create properly structured resources for ipfs_accelerate_py.init_endpoints
    resources["structured_resources"] = {
        "tokenizer": resources["tokenizer"],
        "endpoint_handler": resources["endpoint_handler"],
        "queue": {},  # Add queue dictionary - required by init_endpoints
        "queues": {}, # Add queues dictionary
        "batch_sizes": {}, # Add batch_sizes dictionary
        "consumer_tasks": {}, # Add consumer_tasks dictionary
        "caches": {}, # Add caches dictionary
        "endpoints": {
            "local_endpoints": resources["local_endpoints"],
            "api_endpoints": resources.get("tei_endpoints", {}),
            "libp2p_endpoints": resources.get("libp2p_endpoints", {})
        }
    }

    print(f"Starting test for {len(metadata['models'])} models with {len(endpoint_types)} endpoint types")
    
    # Create test instance and run tests
    tester = test_ipfs_accelerate(resources, metadata)
    
    # Run test asynchronously
    print("Running tests...")
    test_results = asyncio.run(tester.__test__(resources, metadata))
    
    # Generate report after test if DuckDB is available
    if HAVE_DUCKDB:
        # Get DB path from arguments or environment
        db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        # Initialize database handler and generate report
        try:
            db_handler = TestResultsDBHandler(db_path=db_path)
            if db_handler.con is not None:
                # Generate report after test using markdown format
                report_path = "test_report.md"
                report_result = db_handler.generate_report(format="markdown", output_file=report_path)
                if report_result:
                    print(f"\nTest report generated: {report_path}")
        except Exception as e:
            print(f"Error generating automatic test report: {e}")
            traceback.print_exc()
    
    print("Test complete")