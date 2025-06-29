#!/usr/bin/env python3
"""
Test script for the Simulation Accuracy and Validation Framework.

This script tests the foundation of the Simulation Accuracy and Validation Framework
by validating the core classes and database schema.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_simulation_validation_foundation")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import core components
try:
    from duckdb_api.simulation_validation.core.base import (
        SimulationResult,
        HardwareResult,
        ValidationResult,
        SimulationAccuracyFramework
    )
    from duckdb_api.simulation_validation.core.schema import SimulationValidationSchema
    HAS_FRAMEWORK = True
except ImportError as e:
    logger.error(f"Error importing Simulation Validation Framework: {e}")
    HAS_FRAMEWORK = False

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    logger.error(f"DuckDB not available. Install with: pip install duckdb")
    HAS_DUCKDB = False


def test_core_classes():
    """Test the core classes of the Simulation Validation Framework."""
    logger.info("Testing core classes...")
    
    # Create a simulation result
    sim_result = SimulationResult(
        model_id="bert-base-uncased",
        hardware_id="nvidia-a100",
        metrics={
            "throughput_items_per_second": 120.5,
            "average_latency_ms": 8.3,
            "memory_peak_mb": 8192,
            "power_consumption_w": 350.0
        },
        batch_size=8,
        precision="fp16",
        simulation_version="sim_v1.2"
    )
    
    # Create a hardware result
    hw_result = HardwareResult(
        model_id="bert-base-uncased",
        hardware_id="nvidia-a100",
        metrics={
            "throughput_items_per_second": 115.2,
            "average_latency_ms": 8.7,
            "memory_peak_mb": 8450,
            "power_consumption_w": 360.0
        },
        batch_size=8,
        precision="fp16",
        hardware_details={"gpu_memory": "40GB", "cuda_cores": 6912},
        test_environment={"driver_version": "470.82.01", "cuda_version": "11.4"}
    )
    
    # Create a validation result with comparison metrics
    metrics_comparison = {
        "throughput_items_per_second": {
            "absolute_error": 5.3,
            "relative_error": 0.044,
            "mape": 4.4,
            "percent_error": 4.4
        },
        "average_latency_ms": {
            "absolute_error": 0.4,
            "relative_error": 0.048,
            "mape": 4.8,
            "percent_error": -4.8
        },
        "memory_peak_mb": {
            "absolute_error": 258,
            "relative_error": 0.031,
            "mape": 3.1,
            "percent_error": -3.1
        },
        "power_consumption_w": {
            "absolute_error": 10.0,
            "relative_error": 0.028,
            "mape": 2.8,
            "percent_error": -2.8
        }
    }
    
    val_result = ValidationResult(
        simulation_result=sim_result,
        hardware_result=hw_result,
        metrics_comparison=metrics_comparison,
        validation_version="val_v1.0"
    )
    
    # Test to_dict and from_dict for SimulationResult
    sim_dict = sim_result.to_dict()
    sim_result2 = SimulationResult.from_dict(sim_dict)
    
    assert sim_result2.model_id == sim_result.model_id
    assert sim_result2.hardware_id == sim_result.hardware_id
    assert sim_result2.metrics["throughput_items_per_second"] == sim_result.metrics["throughput_items_per_second"]
    
    # Test to_dict and from_dict for HardwareResult
    hw_dict = hw_result.to_dict()
    hw_result2 = HardwareResult.from_dict(hw_dict)
    
    assert hw_result2.model_id == hw_result.model_id
    assert hw_result2.hardware_id == hw_result.hardware_id
    assert hw_result2.metrics["throughput_items_per_second"] == hw_result.metrics["throughput_items_per_second"]
    assert hw_result2.hardware_details["gpu_memory"] == hw_result.hardware_details["gpu_memory"]
    
    # Test to_dict and from_dict for ValidationResult
    val_dict = val_result.to_dict()
    val_result2 = ValidationResult.from_dict(val_dict)
    
    assert val_result2.simulation_result.model_id == val_result.simulation_result.model_id
    assert val_result2.hardware_result.model_id == val_result.hardware_result.model_id
    assert val_result2.metrics_comparison["throughput_items_per_second"]["mape"] == val_result.metrics_comparison["throughput_items_per_second"]["mape"]
    
    logger.info("Core classes test completed successfully.")
    return True


def test_database_schema(db_path):
    """Test the database schema for the Simulation Validation Framework."""
    if not HAS_DUCKDB:
        logger.error("DuckDB not available. Skipping database schema test.")
        return False
    
    logger.info(f"Testing database schema with DB at {db_path}...")
    
    try:
        # Connect to DuckDB database
        conn = duckdb.connect(db_path)
        
        # Create tables
        SimulationValidationSchema.create_tables(conn)
        
        # Test that tables were created
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        expected_tables = [
            "simulation_results",
            "hardware_results",
            "validation_results",
            "calibration_history",
            "drift_detection",
            "simulation_parameters"
        ]
        
        for table in expected_tables:
            if table not in table_names:
                logger.error(f"Expected table {table} not found in database")
                return False
        
        # Create a simulation result
        sim_result = SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 120.5,
                "average_latency_ms": 8.3,
                "memory_peak_mb": 8192,
                "power_consumption_w": 350.0
            },
            batch_size=8,
            precision="fp16",
            simulation_version="sim_v1.2"
        )
        
        # Convert to database record
        sim_record = SimulationValidationSchema.simulation_result_to_db_dict(sim_result)
        
        # Insert into database
        placeholders = ", ".join([f":{k}" for k in sim_record.keys()])
        columns = ", ".join(sim_record.keys())
        
        query = f"INSERT INTO simulation_results ({columns}) VALUES ({placeholders})"
        conn.execute(query, sim_record)
        
        # Verify insertion
        result = conn.execute(f"SELECT * FROM simulation_results WHERE id = '{sim_record['id']}'").fetchone()
        assert result is not None
        
        # Clean up (drop tables)
        for table in expected_tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        
        conn.close()
        
        logger.info("Database schema test completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error testing database schema: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the Simulation Accuracy and Validation Framework")
    parser.add_argument("--db-path", default=":memory:",
                        help="Path to DuckDB database (use :memory: for in-memory database)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip database schema test")
    
    args = parser.parse_args()
    
    # Check if framework is available
    if not HAS_FRAMEWORK:
        logger.error("Simulation Validation Framework not available.")
        logger.error("Please check if the framework is properly installed.")
        return 1
    
    # Run core classes test
    if not test_core_classes():
        logger.error("Core classes test failed.")
        return 1
    
    # Run database schema test
    if not args.skip_db:
        if not test_database_schema(args.db_path):
            logger.error("Database schema test failed.")
            return 1
    else:
        logger.info("Skipping database schema test.")
    
    logger.info("All tests completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())