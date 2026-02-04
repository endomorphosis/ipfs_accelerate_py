"""
DuckDB Integration Module for Benchmark Suite

This module provides integration with DuckDB for storing benchmark results.
It implements the necessary adapters and utilities to store, query, and analyze
benchmark data in a structured database format.
"""

import os
import json
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

try:
    import duckdb
except ImportError:
    duckdb = None

logger = logging.getLogger(__name__)

class BenchmarkDBManager:
    """Database manager for benchmark results."""
    
    def __init__(self, db_path: Optional[str] = None, auto_create: bool = True):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the DuckDB database file. If None, use environment variable
                    BENCHMARK_DB_PATH or default to ./benchmark_db.duckdb
            auto_create: Automatically create the database schema if it doesn't exist
        """
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        # Check if DuckDB is available
        if duckdb is None:
            logger.warning("DuckDB is not installed. Database integration is disabled.")
            self.conn = None
            return
            
        try:
            # Connect to the database
            self.conn = duckdb.connect(self.db_path)
            
            # Create schema if needed
            if auto_create:
                self._create_schema_if_needed()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def _create_schema_if_needed(self):
        """Create the database schema if it doesn't exist."""
        if self.conn is None:
            return
            
        try:
            # Check if tables exist
            result = self.conn.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in result]
            
            # Create tables if they don't exist
            if "hardware_platforms" not in tables:
                self._create_hardware_platforms_table()
            
            if "models" not in tables:
                self._create_models_table()
            
            if "test_runs" not in tables:
                self._create_test_runs_table()
                
            if "performance_results" not in tables:
                self._create_performance_results_table()
                
            if "hardware_compatibility" not in tables:
                self._create_hardware_compatibility_table()
            
            # Create views
            self._create_views()
            
            # Create indexes
            self._create_indexes()
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
    
    def _create_hardware_platforms_table(self):
        """Create the hardware_platforms table."""
        self.conn.execute("""
        CREATE TABLE hardware_platforms (
            hardware_id INTEGER PRIMARY KEY,
            hardware_type VARCHAR NOT NULL,
            device_name VARCHAR,
            platform VARCHAR,
            memory_gb FLOAT,
            simulation_mode BOOLEAN DEFAULT FALSE,
            simulation_warning VARCHAR,
            detection_timestamp TIMESTAMP
        )
        """)
    
    def _create_models_table(self):
        """Create the models table."""
        self.conn.execute("""
        CREATE TABLE models (
            model_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_family VARCHAR,
            model_type VARCHAR,
            modality VARCHAR,
            parameters_million FLOAT,
            last_updated TIMESTAMP
        )
        """)
    
    def _create_test_runs_table(self):
        """Create the test_runs table."""
        self.conn.execute("""
        CREATE TABLE test_runs (
            run_id INTEGER PRIMARY KEY,
            test_name VARCHAR NOT NULL,
            test_type VARCHAR,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            execution_time_seconds FLOAT,
            success BOOLEAN,
            metadata VARCHAR
        )
        """)
    
    def _create_performance_results_table(self):
        """Create the performance_results table."""
        self.conn.execute("""
        CREATE TABLE performance_results (
            result_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER,
            hardware_id INTEGER,
            test_case VARCHAR,
            batch_size INTEGER,
            precision VARCHAR,
            throughput_items_per_second FLOAT,
            average_latency_ms FLOAT,
            memory_peak_mb FLOAT,
            simulation_mode BOOLEAN DEFAULT FALSE,
            simulation_details VARCHAR,
            metrics VARCHAR, -- JSON string
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
    
    def _create_hardware_compatibility_table(self):
        """Create the hardware_compatibility table."""
        self.conn.execute("""
        CREATE TABLE hardware_compatibility (
            compatibility_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER,
            hardware_id INTEGER,
            is_compatible BOOLEAN,
            detection_success BOOLEAN,
            initialization_success BOOLEAN,
            error_message VARCHAR,
            error_type VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
    
    def _create_views(self):
        """Create database views for common queries."""
        # Performance results view with joined model and hardware info
        self.conn.execute("""
        CREATE OR REPLACE VIEW performance_results_view AS
        SELECT 
            pr.result_id,
            pr.run_id,
            m.model_id,
            m.model_name,
            m.model_family,
            m.model_type,
            h.hardware_id,
            h.hardware_type,
            h.device_name,
            pr.test_case,
            pr.batch_size,
            pr.precision,
            pr.throughput_items_per_second,
            pr.average_latency_ms,
            pr.memory_peak_mb,
            pr.simulation_mode,
            pr.created_at
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms h ON pr.hardware_id = h.hardware_id
        """)
        
        # Hardware compatibility view
        self.conn.execute("""
        CREATE OR REPLACE VIEW hardware_compatibility_view AS
        SELECT 
            hc.compatibility_id,
            m.model_name,
            m.model_family,
            h.hardware_type,
            hc.is_compatible,
            hc.error_type,
            hc.created_at
        FROM 
            hardware_compatibility hc
        JOIN 
            models m ON hc.model_id = m.model_id
        JOIN 
            hardware_platforms h ON hc.hardware_id = h.hardware_id
        """)
        
        # Latest performance results view
        self.conn.execute("""
        CREATE OR REPLACE VIEW latest_performance_results AS
        WITH latest_results AS (
            SELECT 
                model_id,
                hardware_id,
                test_case,
                batch_size,
                precision,
                MAX(created_at) as latest_date
            FROM 
                performance_results
            GROUP BY 
                model_id, hardware_id, test_case, batch_size, precision
        )
        SELECT 
            pr.*
        FROM 
            performance_results pr
        JOIN 
            latest_results lr ON 
                pr.model_id = lr.model_id AND
                pr.hardware_id = lr.hardware_id AND
                pr.test_case = lr.test_case AND
                pr.batch_size = lr.batch_size AND
                pr.precision = lr.precision AND
                pr.created_at = lr.latest_date
        """)
    
    def _create_indexes(self):
        """Create indexes for performance optimization."""
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models(model_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_models_family ON models(model_family)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hardware_type ON hardware_platforms(hardware_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_model_hardware ON performance_results(model_id, hardware_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_created_at ON performance_results(created_at)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hc_model_hardware ON hardware_compatibility(model_id, hardware_id)")
    
    def store_performance_result(self, 
                               model_name: str, 
                               hardware_type: str, 
                               test_case: str = "inference",
                               batch_size: int = 1, 
                               precision: str = "fp32",
                               throughput: Optional[float] = None,
                               latency_avg: Optional[float] = None, 
                               memory_peak: Optional[float] = None,
                               simulation_mode: bool = False,
                               simulation_details: Optional[str] = None,
                               metrics: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Store a performance benchmark result in the database.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware (cpu, cuda, rocm, etc.)
            test_case: Type of test (inference, training, etc.)
            batch_size: Batch size used for the benchmark
            precision: Precision used (fp32, fp16, int8, etc.)
            throughput: Throughput in items per second
            latency_avg: Average latency in milliseconds
            memory_peak: Peak memory usage in MB
            simulation_mode: Whether this was run in simulation mode
            simulation_details: Additional details for simulation mode
            metrics: Additional metrics as a dictionary
        
        Returns:
            The result_id if successful, None otherwise
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot store result.")
            return None
        
        try:
            # Get or create a test run
            run_id = self._get_or_create_test_run("performance_benchmark", "performance")
            
            # Get or create the model
            model_id = self._get_or_create_model(model_name)
            
            # Get or create the hardware platform
            hardware_id = self._get_or_create_hardware(hardware_type, simulation_mode)
            
            # Convert metrics to JSON string if provided
            metrics_json = json.dumps(metrics) if metrics else None
            
            # Insert performance result
            result = self.conn.execute("""
            INSERT INTO performance_results (
                run_id, model_id, hardware_id, test_case, batch_size, precision,
                throughput_items_per_second, average_latency_ms, memory_peak_mb,
                simulation_mode, simulation_details, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING result_id
            """, (
                run_id, model_id, hardware_id, test_case, batch_size, precision,
                throughput, latency_avg, memory_peak,
                simulation_mode, simulation_details, metrics_json
            )).fetchone()
            
            self.conn.commit()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to store performance result: {e}")
            self.conn.rollback()
            return None
    
    def store_hardware_compatibility(self, 
                                   model_name: str, 
                                   hardware_type: str,
                                   is_compatible: bool,
                                   detection_success: bool = True,
                                   initialization_success: bool = True,
                                   error_message: Optional[str] = None,
                                   error_type: Optional[str] = None) -> Optional[int]:
        """
        Store hardware compatibility information in the database.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware (cpu, cuda, rocm, etc.)
            is_compatible: Whether the model is compatible with the hardware
            detection_success: Whether hardware detection was successful
            initialization_success: Whether model initialization was successful
            error_message: Error message if any
            error_type: Type of error if any
        
        Returns:
            The compatibility_id if successful, None otherwise
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot store compatibility info.")
            return None
        
        try:
            # Get or create a test run
            run_id = self._get_or_create_test_run("hardware_compatibility_test", "compatibility")
            
            # Get or create the model
            model_id = self._get_or_create_model(model_name)
            
            # Get or create the hardware platform
            hardware_id = self._get_or_create_hardware(hardware_type)
            
            # Insert compatibility info
            result = self.conn.execute("""
            INSERT INTO hardware_compatibility (
                run_id, model_id, hardware_id, is_compatible,
                detection_success, initialization_success, error_message, error_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING compatibility_id
            """, (
                run_id, model_id, hardware_id, is_compatible,
                detection_success, initialization_success, error_message, error_type
            )).fetchone()
            
            self.conn.commit()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to store hardware compatibility: {e}")
            self.conn.rollback()
            return None
    
    def _get_or_create_test_run(self, test_name: str, test_type: str) -> int:
        """
        Get or create a test run.
        
        Args:
            test_name: Name of the test
            test_type: Type of test
        
        Returns:
            The run_id
        """
        # Check if a test run already exists for this test name and is not completed
        result = self.conn.execute("""
        SELECT run_id 
        FROM test_runs 
        WHERE test_name = ? AND test_type = ? AND completed_at IS NULL
        """, (test_name, test_type)).fetchone()
        
        if result:
            return result[0]
        
        # Create a new test run
        now = datetime.datetime.now()
        result = self.conn.execute("""
        INSERT INTO test_runs (test_name, test_type, started_at, success)
        VALUES (?, ?, ?, ?)
        RETURNING run_id
        """, (test_name, test_type, now, True)).fetchone()
        
        return result[0]
    
    def _get_or_create_model(self, model_name: str, 
                           model_family: Optional[str] = None, 
                           model_type: Optional[str] = None,
                           modality: Optional[str] = None, 
                           parameters_million: Optional[float] = None) -> int:
        """
        Get or create a model entry.
        
        Args:
            model_name: Name of the model
            model_family: Model family (bert, gpt, etc.)
            model_type: Model type (encoder, decoder, encoder-decoder)
            modality: Modality (text, vision, audio, multimodal)
            parameters_million: Number of parameters in millions
        
        Returns:
            The model_id
        """
        # Derive model family from name if not provided
        if not model_family:
            for family in ["bert", "gpt", "t5", "vit", "clip", "whisper", "llama"]:
                if family in model_name.lower():
                    model_family = family
                    break
        
        # Check if model exists
        result = self.conn.execute("SELECT model_id FROM models WHERE model_name = ?", (model_name,)).fetchone()
        
        if result:
            model_id = result[0]
            
            # Update model information if provided
            if any([model_family, model_type, modality, parameters_million]):
                update_fields = []
                update_values = []
                
                if model_family:
                    update_fields.append("model_family = ?")
                    update_values.append(model_family)
                
                if model_type:
                    update_fields.append("model_type = ?")
                    update_values.append(model_type)
                
                if modality:
                    update_fields.append("modality = ?")
                    update_values.append(modality)
                
                if parameters_million:
                    update_fields.append("parameters_million = ?")
                    update_values.append(parameters_million)
                
                update_fields.append("last_updated = ?")
                update_values.append(datetime.datetime.now())
                
                if update_fields:
                    update_query = f"""
                    UPDATE models 
                    SET {', '.join(update_fields)}
                    WHERE model_id = ?
                    """
                    update_values.append(model_id)
                    self.conn.execute(update_query, update_values)
            
            return model_id
        
        # Create new model
        now = datetime.datetime.now()
        result = self.conn.execute("""
        INSERT INTO models (model_name, model_family, model_type, modality, parameters_million, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        RETURNING model_id
        """, (model_name, model_family, model_type, modality, parameters_million, now)).fetchone()
        
        return result[0]
    
    def _get_or_create_hardware(self, hardware_type: str, 
                              simulation_mode: bool = False,
                              device_name: Optional[str] = None, 
                              platform: Optional[str] = None,
                              memory_gb: Optional[float] = None) -> int:
        """
        Get or create a hardware platform entry.
        
        Args:
            hardware_type: Type of hardware (cpu, cuda, rocm, etc.)
            simulation_mode: Whether this is a simulated hardware
            device_name: Name of the device
            platform: Platform (linux, windows, macos)
            memory_gb: Amount of memory in GB
        
        Returns:
            The hardware_id
        """
        # Check if hardware exists
        result = self.conn.execute("""
        SELECT hardware_id 
        FROM hardware_platforms 
        WHERE hardware_type = ? AND (simulation_mode = ? OR simulation_mode IS NULL)
        """, (hardware_type, simulation_mode)).fetchone()
        
        if result:
            hardware_id = result[0]
            
            # Update hardware information if provided
            if any([device_name, platform, memory_gb]):
                update_fields = []
                update_values = []
                
                if device_name:
                    update_fields.append("device_name = ?")
                    update_values.append(device_name)
                
                if platform:
                    update_fields.append("platform = ?")
                    update_values.append(platform)
                
                if memory_gb:
                    update_fields.append("memory_gb = ?")
                    update_values.append(memory_gb)
                
                update_fields.append("detection_timestamp = ?")
                update_values.append(datetime.datetime.now())
                
                if update_fields:
                    update_query = f"""
                    UPDATE hardware_platforms 
                    SET {', '.join(update_fields)}
                    WHERE hardware_id = ?
                    """
                    update_values.append(hardware_id)
                    self.conn.execute(update_query, update_values)
            
            return hardware_id
        
        # Create new hardware entry
        now = datetime.datetime.now()
        result = self.conn.execute("""
        INSERT INTO hardware_platforms (
            hardware_type, simulation_mode, device_name, platform, memory_gb, detection_timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
        RETURNING hardware_id
        """, (hardware_type, simulation_mode, device_name, platform, memory_gb, now)).fetchone()
        
        return result[0]
    
    def complete_test_run(self, run_id: int, success: bool = True) -> bool:
        """
        Mark a test run as completed.
        
        Args:
            run_id: ID of the test run
            success: Whether the test run was successful
        
        Returns:
            True if successful, False otherwise
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot complete test run.")
            return False
        
        try:
            now = datetime.datetime.now()
            
            # Get the start time
            result = self.conn.execute("SELECT started_at FROM test_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not result:
                logger.error(f"Test run {run_id} not found")
                return False
            
            started_at = result[0]
            execution_time = (now - started_at).total_seconds() if started_at else None
            
            # Update test run
            self.conn.execute("""
            UPDATE test_runs 
            SET completed_at = ?, success = ?, execution_time_seconds = ?
            WHERE run_id = ?
            """, (now, success, execution_time, run_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to complete test run: {e}")
            self.conn.rollback()
            return False
    
    def get_performance_metrics(self, 
                              model_name: Optional[str] = None, 
                              hardware_type: Optional[str] = None,
                              test_case: Optional[str] = None,
                              batch_size: Optional[int] = None,
                              precision: Optional[str] = None,
                              exclude_simulation: bool = True,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get performance metrics from the database.
        
        Args:
            model_name: Filter by model name
            hardware_type: Filter by hardware type
            test_case: Filter by test case
            batch_size: Filter by batch size
            precision: Filter by precision
            exclude_simulation: Whether to exclude simulation results
            limit: Maximum number of results to return
        
        Returns:
            List of performance metrics
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot get performance metrics.")
            return []
        
        try:
            query = """
            SELECT 
                m.model_name,
                m.model_family,
                h.hardware_type,
                h.device_name,
                pr.test_case,
                pr.batch_size,
                pr.precision,
                pr.throughput_items_per_second,
                pr.average_latency_ms,
                pr.memory_peak_mb,
                pr.simulation_mode,
                pr.created_at
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE 1=1
            """
            
            params = []
            
            if model_name:
                query += " AND m.model_name = ?"
                params.append(model_name)
            
            if hardware_type:
                query += " AND h.hardware_type = ?"
                params.append(hardware_type)
            
            if test_case:
                query += " AND pr.test_case = ?"
                params.append(test_case)
            
            if batch_size:
                query += " AND pr.batch_size = ?"
                params.append(batch_size)
            
            if precision:
                query += " AND pr.precision = ?"
                params.append(precision)
            
            if exclude_simulation:
                query += " AND (pr.simulation_mode = FALSE OR pr.simulation_mode IS NULL)"
            
            query += " ORDER BY pr.created_at DESC LIMIT ?"
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [
                "model_name", "model_family", "hardware_type", "device_name", 
                "test_case", "batch_size", "precision", "throughput_items_per_second",
                "average_latency_ms", "memory_peak_mb", "simulation_mode", "created_at"
            ]
            
            metrics = []
            for row in result:
                metrics.append(dict(zip(columns, row)))
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return []
    
    def get_hardware_compatibility(self, 
                                 model_name: Optional[str] = None, 
                                 hardware_type: Optional[str] = None,
                                 compatibility_filter: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get hardware compatibility information from the database.
        
        Args:
            model_name: Filter by model name
            hardware_type: Filter by hardware type
            compatibility_filter: Filter by compatibility (True, False, or None for all)
        
        Returns:
            List of compatibility information
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot get hardware compatibility.")
            return []
        
        try:
            query = """
            SELECT 
                m.model_name,
                m.model_family,
                h.hardware_type,
                h.device_name,
                hc.is_compatible,
                hc.detection_success,
                hc.initialization_success,
                hc.error_message,
                hc.error_type,
                hc.created_at
            FROM 
                hardware_compatibility hc
            JOIN 
                models m ON hc.model_id = m.model_id
            JOIN 
                hardware_platforms h ON hc.hardware_id = h.hardware_id
            WHERE 1=1
            """
            
            params = []
            
            if model_name:
                query += " AND m.model_name = ?"
                params.append(model_name)
            
            if hardware_type:
                query += " AND h.hardware_type = ?"
                params.append(hardware_type)
            
            if compatibility_filter is not None:
                query += " AND hc.is_compatible = ?"
                params.append(compatibility_filter)
            
            query += " ORDER BY hc.created_at DESC"
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [
                "model_name", "model_family", "hardware_type", "device_name", 
                "is_compatible", "detection_success", "initialization_success",
                "error_message", "error_type", "created_at"
            ]
            
            compatibility = []
            for row in result:
                compatibility.append(dict(zip(columns, row)))
            
            return compatibility
        except Exception as e:
            logger.error(f"Failed to get hardware compatibility: {e}")
            return []
    
    def generate_compatibility_matrix(self, 
                                    exclude_simulation: bool = True) -> Dict[str, Dict[str, bool]]:
        """
        Generate a compatibility matrix showing which models work with which hardware.
        
        Args:
            exclude_simulation: Whether to exclude simulation results
        
        Returns:
            A dictionary mapping model names to a dictionary of hardware types and compatibility
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot generate compatibility matrix.")
            return {}
        
        try:
            query = """
            WITH latest_compatibility AS (
                SELECT 
                    model_id,
                    hardware_id,
                    is_compatible,
                    ROW_NUMBER() OVER (
                        PARTITION BY model_id, hardware_id 
                        ORDER BY created_at DESC
                    ) as row_num
                FROM 
                    hardware_compatibility
            )
            SELECT 
                m.model_name,
                h.hardware_type,
                lc.is_compatible
            FROM 
                latest_compatibility lc
            JOIN 
                models m ON lc.model_id = m.model_id
            JOIN 
                hardware_platforms h ON lc.hardware_id = h.hardware_id
            WHERE 
                lc.row_num = 1
            """
            
            if exclude_simulation:
                query += " AND (h.simulation_mode = FALSE OR h.simulation_mode IS NULL)"
            
            result = self.conn.execute(query).fetchall()
            
            # Build compatibility matrix
            matrix = {}
            for row in result:
                model_name, hardware_type, is_compatible = row
                
                if model_name not in matrix:
                    matrix[model_name] = {}
                
                matrix[model_name][hardware_type] = is_compatible
            
            return matrix
        except Exception as e:
            logger.error(f"Failed to generate compatibility matrix: {e}")
            return {}
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
        
        Returns:
            Result of the query as a list of dictionaries
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot execute query.")
            return []
        
        try:
            params = params or []
            result = self.conn.execute(query, params).fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in self.conn.description]
            
            # Convert to list of dictionaries
            rows = []
            for row in result:
                rows.append(dict(zip(column_names, row)))
            
            return rows
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []
    
    def optimize_database(self) -> bool:
        """
        Optimize the database for better performance.
        
        Returns:
            True if successful, False otherwise
        """
        if self.conn is None:
            logger.warning("Database connection not available. Cannot optimize database.")
            return False
        
        try:
            # Analyze tables for optimization
            self.conn.execute("PRAGMA analyze")
            
            # Run VACUUM to reclaim space
            self.conn.execute("VACUUM")
            
            # Update statistics
            self.conn.execute("ANALYZE")
            
            return True
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

# Helper class to automatically handle database connections
class BenchmarkDBContext:
    """Context manager for database operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the context manager.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.db = None
    
    def __enter__(self) -> BenchmarkDBManager:
        """Enter the context and return the database manager."""
        self.db = BenchmarkDBManager(self.db_path)
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and close the database connection."""
        if self.db is not None:
            self.db.close()


# Function to get a database manager with environment variable support
def get_db_manager(db_path: Optional[str] = None) -> BenchmarkDBManager:
    """
    Get a database manager instance with environment variable support.
    
    Args:
        db_path: Path to the DuckDB database file
    
    Returns:
        A BenchmarkDBManager instance
    """
    db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    return BenchmarkDBManager(db_path)


# Utility functions for working with benchmark results
def store_benchmark_result(result: Dict[str, Any], db_path: Optional[str] = None) -> Optional[int]:
    """
    Store a benchmark result in the database.
    
    Args:
        result: Benchmark result dictionary
        db_path: Path to the DuckDB database file
    
    Returns:
        The result_id if successful, None otherwise
    """
    with BenchmarkDBContext(db_path) as db:
        if db.conn is None:
            logger.warning("Database connection not available. Cannot store benchmark result.")
            return None
        
        # Extract required fields
        model_name = result.get("model_name")
        hardware_type = result.get("hardware_type")
        test_case = result.get("test_case", "inference")
        batch_size = result.get("batch_size", 1)
        precision = result.get("precision", "fp32")
        throughput = result.get("throughput_items_per_second")
        latency_avg = result.get("average_latency_ms")
        memory_peak = result.get("memory_peak_mb")
        simulation_mode = result.get("simulation_mode", False)
        simulation_details = result.get("simulation_details")
        
        # Extract additional metrics
        metrics = {k: v for k, v in result.items() if k not in [
            "model_name", "hardware_type", "test_case", "batch_size", "precision",
            "throughput_items_per_second", "average_latency_ms", "memory_peak_mb",
            "simulation_mode", "simulation_details"
        ]}
        
        return db.store_performance_result(
            model_name=model_name,
            hardware_type=hardware_type,
            test_case=test_case,
            batch_size=batch_size,
            precision=precision,
            throughput=throughput,
            latency_avg=latency_avg,
            memory_peak=memory_peak,
            simulation_mode=simulation_mode,
            simulation_details=simulation_details,
            metrics=metrics
        )