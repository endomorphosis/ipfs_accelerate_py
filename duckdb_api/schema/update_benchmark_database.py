#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update benchmark database with real hardware benchmark results
"""

import os
import sys
import json
import logging
import argparse
import duckdb
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_or_update_schema(db_path: str) -> bool:
    """
    Create or update the database schema.
    
    Args:
        db_path: Path to the DuckDB database
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating or updating schema in {db_path}...")
    
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Get existing table schemas to check column names
        try:
            # Sample a row from models table to see column names
            conn.execute("SELECT * FROM models LIMIT 1")
            has_models_table = True
        except:
            has_models_table = False
            
        if not has_models_table:
            # Create models table
            conn.execute("""
            CREATE TABLE models (
                model_id INTEGER PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_family VARCHAR,
                model_type VARCHAR,
                model_size VARCHAR,
                parameter_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
            """)
            logger.info("Created models table")
        
        # Check for hardware_platforms table
        try:
            conn.execute("SELECT * FROM hardware_platforms LIMIT 1")
            has_hardware_table = True
        except:
            has_hardware_table = False
            
        if not has_hardware_table:
            # Create hardware_platforms table
            conn.execute("""
            CREATE TABLE hardware_platforms (
                hardware_id INTEGER PRIMARY KEY,
                hardware_type VARCHAR NOT NULL,
                hardware_name VARCHAR,
                is_simulated BOOLEAN DEFAULT FALSE,
                simulation_reason VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
            """)
            logger.info("Created hardware_platforms table")
        
        # Check for test_runs table
        try:
            conn.execute("SELECT * FROM test_runs LIMIT 1")
            has_test_runs_table = True
        except:
            has_test_runs_table = False
            
        if not has_test_runs_table:
            # Create test_runs table
            conn.execute("""
            CREATE TABLE test_runs (
                run_id INTEGER PRIMARY KEY,
                test_name VARCHAR NOT NULL,
                test_type VARCHAR NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status VARCHAR DEFAULT 'running',
                metadata JSON
            )
            """)
            logger.info("Created test_runs table")
        
        # Check for performance_results table
        try:
            conn.execute("SELECT * FROM performance_results LIMIT 1")
            has_performance_table = True
        except:
            has_performance_table = False
            
        if not has_performance_table:
            # Create performance_results table
            conn.execute("""
            CREATE TABLE performance_results (
                performance_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_id INTEGER,
                hardware_id INTEGER,
                batch_size INTEGER DEFAULT 1,
                throughput_items_per_second FLOAT,
                latency_ms FLOAT,
                memory_mb FLOAT,
                is_simulated BOOLEAN DEFAULT FALSE,
                simulation_reason VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
            )
            """)
            logger.info("Created performance_results table")
        
        # Close connection
        conn.close()
        logger.info("Schema creation/update complete")
        return True
    except Exception as e:
        logger.error(f"Error creating/updating schema: {e}")
        return False

def get_or_create_model(conn, model_name: str, model_family: Optional[str] = None,
                     model_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Get or create a model in the database.
    
    Args:
        conn: Database connection
        model_name: Name of the model
        model_family: Family of the model
        model_type: Type of the model
        metadata: Additional metadata
        
    Returns:
        Model ID if successful, None otherwise
    """
    try:
        # Check if model exists
        query = "SELECT model_id FROM models WHERE model_name = ?"
        result = conn.execute(query, [model_name]).fetchone()
        
        if result:
            # Model exists
            model_id = result[0]
            logger.info(f"Found existing model {model_name} with ID {model_id}")
            return model_id
        
        # Create new model
        metadata_json = json.dumps(metadata or {})
        query = """
        INSERT INTO models (model_name, model_family, model_type, metadata)
        VALUES (?, ?, ?, ?)
        RETURNING model_id
        """
        model_id = conn.execute(query, [model_name, model_family, model_type, metadata_json]).fetchone()[0]
        logger.info(f"Created new model {model_name} with ID {model_id}")
        return model_id
    except Exception as e:
        logger.error(f"Error getting or creating model: {e}")
        return None

def get_or_create_hardware(conn, hardware_type: str, hardware_name: Optional[str] = None,
                        is_simulated: bool = False, simulation_reason: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Get or create a hardware platform in the database.
    
    Args:
        conn: Database connection
        hardware_type: Type of hardware
        hardware_name: Name of hardware
        is_simulated: Whether hardware is simulated
        simulation_reason: Reason for simulation
        metadata: Additional metadata
        
    Returns:
        Hardware ID if successful, None otherwise
    """
    try:
        # Check if hardware exists
        query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?"
        result = conn.execute(query, [hardware_type]).fetchone()
        
        if result:
            # Hardware exists
            hardware_id = result[0]
            logger.info(f"Found existing hardware {hardware_type} with ID {hardware_id}")
            
            # Update simulation status if necessary
            query = """
            UPDATE hardware_platforms
            SET is_simulated = ?, simulation_reason = ?
            WHERE hardware_id = ?
            """
            conn.execute(query, [is_simulated, simulation_reason, hardware_id])
            
            return hardware_id
        
        # Create new hardware
        metadata_json = json.dumps(metadata or {})
        query = """
        INSERT INTO hardware_platforms (hardware_type, hardware_name, is_simulated, simulation_reason, metadata)
        VALUES (?, ?, ?, ?, ?)
        RETURNING hardware_id
        """
        hardware_id = conn.execute(query, [hardware_type, hardware_name, is_simulated, simulation_reason, metadata_json]).fetchone()[0]
        logger.info(f"Created new hardware {hardware_type} with ID {hardware_id}")
        return hardware_id
    except Exception as e:
        logger.error(f"Error getting or creating hardware: {e}")
        return None

def create_test_run(conn, test_name: str, test_type: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Create a test run in the database.
    
    Args:
        conn: Database connection
        test_name: Name of the test
        test_type: Type of the test
        metadata: Additional metadata
        
    Returns:
        Run ID if successful, None otherwise
    """
    try:
        # Create new test run
        metadata_json = json.dumps(metadata or {})
        query = """
        INSERT INTO test_runs (test_name, test_type, metadata)
        VALUES (?, ?, ?)
        RETURNING run_id
        """
        run_id = conn.execute(query, [test_name, test_type, metadata_json]).fetchone()[0]
        logger.info(f"Created new test run with ID {run_id}")
        return run_id
    except Exception as e:
        logger.error(f"Error creating test run: {e}")
        return None

def complete_test_run(conn, run_id: int, status: str = "completed") -> bool:
    """
    Complete a test run in the database.
    
    Args:
        conn: Database connection
        run_id: ID of the test run
        status: Status of the test run
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Update test run
        query = """
        UPDATE test_runs
        SET end_time = CURRENT_TIMESTAMP, status = ?
        WHERE run_id = ?
        """
        conn.execute(query, [status, run_id])
        logger.info(f"Completed test run with ID {run_id}")
        return True
    except Exception as e:
        logger.error(f"Error completing test run: {e}")
        return False

def store_performance_result(conn, run_id: int, model_id: int, hardware_id: int, batch_size: int,
                          throughput: float, latency: float, memory: Optional[float] = None,
                          is_simulated: bool = False, simulation_reason: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Store a performance result in the database.
    
    Args:
        conn: Database connection
        run_id: ID of the test run
        model_id: ID of the model
        hardware_id: ID of the hardware
        batch_size: Batch size used
        throughput: Throughput in items per second
        latency: Latency in milliseconds
        memory: Memory usage in megabytes
        is_simulated: Whether result is simulated
        simulation_reason: Reason for simulation
        metadata: Additional metadata
        
    Returns:
        Performance ID if successful, None otherwise
    """
    try:
        # Store performance result
        metadata_json = json.dumps(metadata or {})
        query = """
        INSERT INTO performance_results (
            run_id, model_id, hardware_id, batch_size, throughput_items_per_second,
            latency_ms, memory_mb, is_simulated, simulation_reason, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING performance_id
        """
        params = [
            run_id, model_id, hardware_id, batch_size, throughput,
            latency, memory, is_simulated, simulation_reason, metadata_json
        ]
        performance_id = conn.execute(query, params).fetchone()[0]
        logger.info(f"Stored performance result with ID {performance_id}")
        return performance_id
    except Exception as e:
        logger.error(f"Error storing performance result: {e}")
        return None

def import_benchmark_results(json_path: str, db_path: str) -> bool:
    """
    Import benchmark results from a JSON file into the database.
    
    Args:
        json_path: Path to the JSON file
        db_path: Path to the database
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Importing benchmark results from {json_path} into {db_path}...")
    
    try:
        # Load JSON file
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Create or update schema
        if not create_or_update_schema(db_path):
            logger.error("Failed to create or update schema")
            return False
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Create test run
        run_id = create_test_run(
            conn=conn,
            test_name="bert_benchmark",
            test_type="benchmark",
            metadata={"source": "bert_benchmark_script", "timestamp": data.get("timestamp")}
        )
        
        if run_id is None:
            logger.error("Failed to create test run")
            conn.close()
            return False
        
        # Process CPU results
        if "cpu" in data:
            cpu_result = data["cpu"]
            if cpu_result.get("success", False):
                # Get or create model
                model_id = get_or_create_model(
                    conn=conn,
                    model_name=cpu_result.get("model_name", "prajjwal1/bert-tiny"),
                    model_family="embedding",
                    model_type="text",
                    metadata={"source": "bert_benchmark_script"}
                )
                
                # Get or create hardware
                hardware_id = get_or_create_hardware(
                    conn=conn,
                    hardware_type="cpu",
                    hardware_name="CPU",
                    is_simulated=cpu_result.get("is_simulated", False),
                    simulation_reason=cpu_result.get("simulation_reason")
                )
                
                # Store performance result
                latency = cpu_result.get("latency_ms", {}).get("mean") if isinstance(cpu_result.get("latency_ms"), dict) else cpu_result.get("latency_ms")
                performance_id = store_performance_result(
                    conn=conn,
                    run_id=run_id,
                    model_id=model_id,
                    hardware_id=hardware_id,
                    batch_size=cpu_result.get("batch_size", 1),
                    throughput=cpu_result.get("throughput_items_per_second", 0.0),
                    latency=latency,
                    memory=cpu_result.get("memory_mb"),
                    is_simulated=cpu_result.get("is_simulated", False),
                    simulation_reason=cpu_result.get("simulation_reason"),
                    metadata={"raw_result": cpu_result}
                )
                
                logger.info(f"Imported CPU benchmark result with ID {performance_id}")
        
        # Process CUDA results
        if "cuda" in data:
            cuda_result = data["cuda"]
            if cuda_result.get("success", False):
                # Get or create model
                model_id = get_or_create_model(
                    conn=conn,
                    model_name=cuda_result.get("model_name", "prajjwal1/bert-tiny"),
                    model_family="embedding",
                    model_type="text",
                    metadata={"source": "bert_benchmark_script"}
                )
                
                # Get or create hardware
                hardware_id = get_or_create_hardware(
                    conn=conn,
                    hardware_type="cuda",
                    hardware_name="CUDA",
                    is_simulated=cuda_result.get("is_simulated", False),
                    simulation_reason=cuda_result.get("simulation_reason")
                )
                
                # Store performance result
                latency = cuda_result.get("latency_ms", {}).get("mean") if isinstance(cuda_result.get("latency_ms"), dict) else cuda_result.get("latency_ms")
                performance_id = store_performance_result(
                    conn=conn,
                    run_id=run_id,
                    model_id=model_id,
                    hardware_id=hardware_id,
                    batch_size=cuda_result.get("batch_size", 1),
                    throughput=cuda_result.get("throughput_items_per_second", 0.0),
                    latency=latency,
                    memory=cuda_result.get("memory_mb"),
                    is_simulated=cuda_result.get("is_simulated", False),
                    simulation_reason=cuda_result.get("simulation_reason"),
                    metadata={"raw_result": cuda_result}
                )
                
                logger.info(f"Imported CUDA benchmark result with ID {performance_id}")
        
        # Complete test run
        complete_test_run(conn=conn, run_id=run_id)
        
        # Close connection
        conn.close()
        
        logger.info("Import complete")
        return True
    except Exception as e:
        logger.error(f"Error importing benchmark results: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Import benchmark results into database")
    parser.add_argument("--input", type=str, default="bert_benchmark_results.json", help="Path to benchmark results JSON file")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to database")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Import benchmark results
    if import_benchmark_results(args.input, args.db_path):
        logger.info("Successfully imported benchmark results")
        return 0
    else:
        logger.error("Failed to import benchmark results")
        return 1

if __name__ == "__main__":
    sys.exit(main())