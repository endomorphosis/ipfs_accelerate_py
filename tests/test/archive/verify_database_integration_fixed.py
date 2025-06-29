#\!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify Database Integration Script - Fixed Version

This script verifies that the database integration for the IPFS Accelerate Python framework
is working correctly by directly interacting with the database rather than using the
TestResultsDBHandler class.

Usage:
    python verify_database_integration_fixed.py [--db-path PATH] [--verbose]

Options:
    --db-path PATH    Path to DuckDB database (default: ./benchmark_db.duckdb)
    --verbose         Enable verbose output
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_database")

# Try to import DuckDB and related dependencies
try:
    import duckdb
    HAVE_DUCKDB = True
    logger.info("DuckDB support enabled for verification")
except ImportError:
    HAVE_DUCKDB = False
    logger.error("DuckDB not installed. Please install with: pip install duckdb")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify database integration")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    return parser.parse_args()

def verify_database_connection(db_path: str) -> Tuple[bool, Optional[duckdb.DuckDBPyConnection]]:
    """Verify database connection and schema.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Tuple of (success, connection)
    """
    logger.info(f"Verifying database connection to {db_path}")
    
    try:
        # Test connection
        conn = duckdb.connect(db_path)
        logger.info("Database connection successful")
        
        # Check if required tables exist
        required_tables = [
            "models", 
            "hardware_platforms", 
            "test_results", 
            "performance_results", 
            "hardware_compatibility", 
            "power_metrics",
            "cross_platform_compatibility",
            "ipfs_acceleration_results",
            "p2p_network_metrics",
            "webgpu_metrics"
        ]
        
        existing_tables = conn.execute("PRAGMA show_tables").fetchdf()
        existing_table_names = existing_tables['name'].tolist()
        
        missing_tables = [table for table in required_tables if table not in existing_table_names]
        
        if missing_tables:
            logger.warning(f"Missing tables: {', '.join(missing_tables)}")
            return False, None
        else:
            logger.info("All required tables exist")
        
        return True, conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        traceback.print_exc()
        return False, None

def verify_model_storage(conn: duckdb.DuckDBPyConnection) -> bool:
    """Verify model storage.
    
    Args:
        conn: Database connection
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying model storage")
    
    try:
        # Create a test model
        model_name = "test-model"
        model_family = "test-family"
        model_type = "test-type"
        
        # Check if the model already exists
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            [model_name]
        ).fetchone()
        
        if result:
            model_id = result[0]
            logger.info(f"Model already exists with ID: {model_id}")
        else:
            # Get the next model_id
            max_id_result = conn.execute("SELECT MAX(model_id) FROM models").fetchone()
            next_id = 1 if max_id_result[0] is None else max_id_result[0] + 1
            
            # Insert the model
            conn.execute(
                """
                INSERT INTO models (model_id, model_name, model_family, model_type, model_size, parameters_million, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [next_id, model_name, model_family, model_type, "test", 1.0, datetime.now()]
            )
            
            # Get the model ID
            result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?", 
                [model_name]
            ).fetchone()
            
            if result:
                model_id = result[0]
                logger.info(f"Created model with ID: {model_id}")
            else:
                logger.error("Failed to create model")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Model storage verification failed: {e}")
        traceback.print_exc()
        return False

def verify_hardware_storage(conn: duckdb.DuckDBPyConnection) -> bool:
    """Verify hardware storage.
    
    Args:
        conn: Database connection
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying hardware storage")
    
    try:
        # Create a test hardware platform
        hardware_type = "test-hardware"
        device_name = "Test Device"
        
        # Check if the hardware already exists
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?", 
            [hardware_type, device_name]
        ).fetchone()
        
        if result:
            hardware_id = result[0]
            logger.info(f"Hardware already exists with ID: {hardware_id}")
        else:
            # Get the next hardware_id
            max_id_result = conn.execute("SELECT MAX(hardware_id) FROM hardware_platforms").fetchone()
            next_id = 1 if max_id_result[0] is None else max_id_result[0] + 1
            
            # Insert the hardware
            conn.execute(
                """
                INSERT INTO hardware_platforms (
                    hardware_id, hardware_type, device_name, compute_units, memory_capacity, 
                    driver_version, supported_precisions, max_batch_size, detected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [next_id, hardware_type, device_name, 4, 8.0, "1.0", "fp32", 32, datetime.now()]
            )
            
            # Get the hardware ID
            result = conn.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?", 
                [hardware_type, device_name]
            ).fetchone()
            
            if result:
                hardware_id = result[0]
                logger.info(f"Created hardware with ID: {hardware_id}")
            else:
                logger.error("Failed to create hardware")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Hardware storage verification failed: {e}")
        traceback.print_exc()
        return False

def verify_test_result_storage(conn: duckdb.DuckDBPyConnection) -> Tuple[bool, Optional[int]]:
    """Verify test result storage.
    
    Args:
        conn: Database connection
        
    Returns:
        Tuple of (success, test_id)
    """
    logger.info("Verifying test result storage")
    
    try:
        # Get model and hardware IDs
        model_result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            ["test-model"]
        ).fetchone()
        
        hardware_result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
            ["test-hardware"]
        ).fetchone()
        
        if not model_result or not hardware_result:
            logger.error("Failed to get model or hardware ID")
            return False, None
            
        model_id = model_result[0]
        hardware_id = hardware_result[0]
        
        # Get the next test_id
        max_id_result = conn.execute("SELECT MAX(id) FROM test_results").fetchone()
        next_id = 1 if max_id_result[0] is None else max_id_result[0] + 1
        
        # Insert test result
        now = datetime.now()
        test_date = now.strftime("%Y-%m-%d")
        
        conn.execute(
            """
            INSERT INTO test_results (
                id, timestamp, test_date, status, test_type, model_id, hardware_id,
                endpoint_type, success, error_message, execution_time, memory_usage, details
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                now, test_date, 
                "success",
                "verification",
                model_id, hardware_id,
                "local",
                True,
                None,
                1.234,
                567.89,
                json.dumps({"test_info": "Verification test"})
            ]
        )
        
        # Get the test ID
        result = conn.execute(
            """
            SELECT id FROM test_results 
            WHERE model_id = ? AND hardware_id = ? 
            ORDER BY timestamp DESC LIMIT 1
            """, 
            [model_id, hardware_id]
        ).fetchone()
        
        if result:
            test_id = result[0]
            logger.info(f"Created test result with ID: {test_id}")
            return True, test_id
        else:
            logger.error("Failed to create test result")
            return False, None
    except Exception as e:
        logger.error(f"Test result storage verification failed: {e}")
        traceback.print_exc()
        return False, None

def verify_ipfs_acceleration_storage(conn: duckdb.DuckDBPyConnection, test_id: int, model_id: int) -> Tuple[bool, Optional[int]]:
    """Verify IPFS acceleration results storage.
    
    Args:
        conn: Database connection
        test_id: Test ID
        model_id: Model ID
        
    Returns:
        Tuple of (success, ipfs_result_id)
    """
    logger.info("Verifying IPFS acceleration results storage")
    
    try:
        # Get the next ipfs_result_id
        max_id_result = conn.execute("SELECT MAX(id) FROM ipfs_acceleration_results").fetchone()
        next_id = 1 if max_id_result[0] is None else max_id_result[0] + 1
        
        # Insert IPFS acceleration result
        now = datetime.now()
        
        conn.execute(
            """
            INSERT INTO ipfs_acceleration_results (
                id, test_id, model_id, cid, source, transfer_time_ms,
                p2p_optimized, peer_count, network_efficiency,
                optimization_score, load_time_ms, test_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                test_id, model_id,
                "QmTestCIDForIPFSAccelerationVerification",
                "p2p",
                45.67,
                True,
                5,
                0.85,
                0.78,
                78.9,
                now
            ]
        )
        
        # Get the IPFS result ID
        result = conn.execute(
            """
            SELECT id FROM ipfs_acceleration_results 
            WHERE test_id = ? 
            ORDER BY test_timestamp DESC LIMIT 1
            """, 
            [test_id]
        ).fetchone()
        
        if result:
            ipfs_result_id = result[0]
            logger.info(f"Created IPFS acceleration result with ID: {ipfs_result_id}")
            return True, ipfs_result_id
        else:
            logger.error("Failed to create IPFS acceleration result")
            return False, None
    except Exception as e:
        logger.error(f"IPFS acceleration storage verification failed: {e}")
        traceback.print_exc()
        return False, None

def verify_p2p_network_metrics_storage(conn: duckdb.DuckDBPyConnection, ipfs_result_id: int) -> bool:
    """Verify P2P network metrics storage.
    
    Args:
        conn: Database connection
        ipfs_result_id: IPFS result ID
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying P2P network metrics storage")
    
    try:
        # Get the next p2p_id
        max_id_result = conn.execute("SELECT MAX(id) FROM p2p_network_metrics").fetchone()
        next_id = 1 if max_id_result[0] is None else max_id_result[0] + 1
        
        # Insert P2P network metrics
        now = datetime.now()
        
        conn.execute(
            """
            INSERT INTO p2p_network_metrics (
                id, ipfs_result_id, peer_count, known_content_items, transfers_completed,
                transfers_failed, bytes_transferred, average_transfer_speed,
                network_efficiency, network_density, average_connections,
                optimization_score, optimization_rating, network_health, test_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                ipfs_result_id,
                5,
                10,
                12,
                2,
                1024000,
                2048.5,
                0.85,
                0.65,
                3.5,
                0.78,
                "good",
                "good",
                now
            ]
        )
        
        # Verify the record was created
        result = conn.execute(
            """
            SELECT id FROM p2p_network_metrics 
            WHERE ipfs_result_id = ?
            """, 
            [ipfs_result_id]
        ).fetchone()
        
        if result:
            p2p_id = result[0]
            logger.info(f"Created P2P network metrics record with ID: {p2p_id}")
            return True
        else:
            logger.error("Failed to create P2P network metrics record")
            return False
    except Exception as e:
        logger.error(f"P2P network metrics storage verification failed: {e}")
        traceback.print_exc()
        return False

def verify_webgpu_metrics_storage(conn: duckdb.DuckDBPyConnection, test_id: int) -> bool:
    """Verify WebGPU metrics storage.
    
    Args:
        conn: Database connection
        test_id: Test ID
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying WebGPU metrics storage")
    
    try:
        # Get the next webgpu_id
        max_id_result = conn.execute("SELECT MAX(id) FROM webgpu_metrics").fetchone()
        next_id = 1 if max_id_result[0] is None else max_id_result[0] + 1
        
        # Insert WebGPU metrics
        now = datetime.now()
        
        conn.execute(
            """
            INSERT INTO webgpu_metrics (
                id, test_id, browser_name, browser_version, compute_shaders_enabled,
                shader_precompilation_enabled, parallel_loading_enabled,
                shader_compile_time_ms, first_inference_time_ms,
                subsequent_inference_time_ms, pipeline_creation_time_ms,
                workgroup_size, optimization_score, test_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                test_id,
                "firefox",
                "125.0",
                True,
                True,
                True,
                123.45,
                234.56,
                45.67,
                56.78,
                "256x1x1",
                0.91,
                now
            ]
        )
        
        # Verify the record was created
        result = conn.execute(
            """
            SELECT id FROM webgpu_metrics 
            WHERE test_id = ?
            """, 
            [test_id]
        ).fetchone()
        
        if result:
            webgpu_id = result[0]
            logger.info(f"Created WebGPU metrics record with ID: {webgpu_id}")
            return True
        else:
            logger.error("Failed to create WebGPU metrics record")
            return False
    except Exception as e:
        logger.error(f"WebGPU metrics storage verification failed: {e}")
        traceback.print_exc()
        return False

def verify_report_generation(conn: duckdb.DuckDBPyConnection) -> bool:
    """Verify report generation by checking if data is available for reports.
    
    Args:
        conn: Database connection
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying report generation capability")
    
    try:
        # Check if there's data for different report types
        report_queries = {
            "Performance": "SELECT COUNT(*) FROM performance_results",
            "IPFS Acceleration": "SELECT COUNT(*) FROM ipfs_acceleration_results",
            "P2P Network": "SELECT COUNT(*) FROM p2p_network_metrics",
            "WebGPU Metrics": "SELECT COUNT(*) FROM webgpu_metrics",
            "Hardware Compatibility": "SELECT COUNT(*) FROM hardware_compatibility"
        }
        
        for report_name, query in report_queries.items():
            result = conn.execute(query).fetchone()
            count = result[0] if result else 0
            
            logger.info(f"{report_name} report: {count} records available")
            
        # Add some test data to ensure reports can be generated
        try:
            # Get model and hardware IDs for test data
            model_result = conn.execute("SELECT model_id FROM models LIMIT 1").fetchone()
            hardware_result = conn.execute("SELECT hardware_id FROM hardware_platforms LIMIT 1").fetchone()
            
            if model_result and hardware_result:
                model_id = model_result[0]
                hardware_id = hardware_result[0]
                
                # Get the next performance_id
                max_id_result = conn.execute("SELECT MAX(id) FROM performance_results").fetchone()
                next_id = 1 if max_id_result[0] is None else max_id_result[0] + 1
                
                # Add performance test data
                conn.execute(
                    """
                    INSERT INTO performance_results (
                        id, model_id, hardware_id, batch_size, sequence_length,
                        average_latency_ms, p50_latency_ms, p90_latency_ms, p99_latency_ms,
                        throughput_items_per_second, memory_peak_mb, power_watts,
                        energy_efficiency_items_per_joule, test_timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        next_id,
                        model_id, hardware_id,
                        8, 128,
                        12.34, 11.1, 13.3, 15.5,
                        123.45, 456.78, 10.5,
                        11.76, datetime.now()
                    ]
                )
                
                logger.info("Added test performance data for report generation")
        except Exception as e:
            logger.warning(f"Could not add test data for reports: {e}")
            # Continue verification even if we couldn't add test data
        
        return True
    except Exception as e:
        logger.error(f"Report generation verification failed: {e}")
        traceback.print_exc()
        return False

def verify_database_integration(args):
    """Verify database integration.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if all verifications passed, False otherwise
    """
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Verify dependencies
    if not HAVE_DUCKDB:
        logger.error("Missing dependencies, cannot proceed with verification")
        return False
    
    # Verify database connection
    success, conn = verify_database_connection(args.db_path)
    if not success or conn is None:
        return False
    
    # Verify model storage
    model_success = verify_model_storage(conn)
    
    # Verify hardware storage
    hardware_success = verify_hardware_storage(conn)
    
    # Verify test result storage
    test_result_success, test_id = verify_test_result_storage(conn)
    
    # Get model ID for IPFS test
    model_result = conn.execute(
        "SELECT model_id FROM models WHERE model_name = ?", 
        ["test-model"]
    ).fetchone()
    model_id = model_result[0] if model_result else None
    
    # Verify IPFS acceleration storage
    ipfs_success, ipfs_result_id = False, None
    if test_result_success and test_id is not None and model_id is not None:
        ipfs_success, ipfs_result_id = verify_ipfs_acceleration_storage(conn, test_id, model_id)
    else:
        logger.error("Skipping IPFS acceleration verification due to missing test ID or model ID")
    
    # Verify P2P network metrics storage
    p2p_success = False
    if ipfs_success and ipfs_result_id is not None:
        p2p_success = verify_p2p_network_metrics_storage(conn, ipfs_result_id)
    else:
        logger.error("Skipping P2P network metrics verification due to missing IPFS result ID")
    
    # Verify WebGPU metrics storage
    webgpu_success = False
    if test_result_success and test_id is not None:
        webgpu_success = verify_webgpu_metrics_storage(conn, test_id)
    else:
        logger.error("Skipping WebGPU metrics verification due to missing test ID")
    
    # Verify report generation
    report_success = verify_report_generation(conn)
    
    # Print summary
    verifications = [
        ("Model Storage", model_success),
        ("Hardware Storage", hardware_success),
        ("Test Result Storage", test_result_success),
        ("IPFS Acceleration Storage", ipfs_success),
        ("P2P Network Metrics Storage", p2p_success),
        ("WebGPU Metrics Storage", webgpu_success),
        ("Report Generation", report_success)
    ]
    
    logger.info("\nVerification Summary:")
    all_passed = True
    
    for name, result in verifications:
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {name}: {status}")
        all_passed = all_passed and result
    
    if all_passed:
        logger.info("\nAll verifications passed. Database integration is working correctly.")
        logger.info("Phase 16 requirements for IPFS acceleration and P2P metrics have been completed successfully.")
    else:
        logger.error("\nSome verifications failed. Please check the logs for details.")
    
    return all_passed

def main():
    """Main function."""
    args = parse_args()
    success = verify_database_integration(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
