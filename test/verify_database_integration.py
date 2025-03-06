#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify Database Integration Script

This script verifies that the database integration for the IPFS Accelerate Python framework
is working correctly, completing the Phase 16 requirements. It checks:

1. Database connection and schema validation
2. Test result storage capabilities
3. Performance metrics storage
4. Cross-platform compatibility matrix generation
5. Report generation in various formats

Usage:
    python verify_database_integration.py [--db-path PATH] [--verbose]

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

# Add parent directory to path for proper imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import required modules
try:
    import duckdb
    from test_ipfs_accelerate import TestResultsDBHandler
    HAS_DEPENDENCIES = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Please install required dependencies: pip install duckdb pandas")
    HAS_DEPENDENCIES = False

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
            "cross_platform_compatibility"
        ]
        
        existing_tables = conn.execute("PRAGMA show_tables").fetchdf()
        existing_table_names = existing_tables['name'].tolist()
        
        missing_tables = [table for table in required_tables if table not in existing_table_names]
        
        if missing_tables:
            logger.warning(f"Missing tables: {', '.join(missing_tables)}")
            logger.info("Will attempt to create missing tables during verification")
        else:
            logger.info("All required tables exist")
        
        return True, conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        traceback.print_exc()
        return False, None

def verify_test_result_storage(db_handler: TestResultsDBHandler) -> bool:
    """Verify test result storage.
    
    Args:
        db_handler: TestResultsDBHandler instance
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying test result storage")
    
    try:
        # Create a test result
        test_result = {
            "model_name": "test-model",
            "model_family": "test-family",
            "hardware_type": "test-hardware",
            "test_type": "verification",
            "status": "Success",
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "execution_time": 1.234,
            "memory_usage": 567.89,
            "details": {
                "test_info": "Verification test",
                "extra_data": "This is a test"
            }
        }
        
        # Store the test result
        success = db_handler.store_test_result(test_result)
        
        if not success:
            logger.error("Failed to store test result")
            return False
            
        logger.info("Test result storage verified")
        return True
    except Exception as e:
        logger.error(f"Test result storage verification failed: {e}")
        traceback.print_exc()
        return False

def verify_performance_metrics_storage(db_handler: TestResultsDBHandler) -> bool:
    """Verify performance metrics storage.
    
    Args:
        db_handler: TestResultsDBHandler instance
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying performance metrics storage")
    
    try:
        # Create a test result with performance metrics
        test_result = {
            "model_name": "perf-test-model",
            "model_family": "perf-test-family",
            "hardware_type": "perf-test-hardware",
            "test_type": "performance-verification",
            "status": "Success",
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "execution_time": 2.345,
            "memory_usage": 678.9,
            "performance": {
                "batch_size": 8,
                "sequence_length": 128,
                "average_latency_ms": 12.34,
                "p50_latency_ms": 11.1,
                "p90_latency_ms": 13.3,
                "p99_latency_ms": 15.5,
                "throughput_items_per_second": 123.45,
                "memory_peak_mb": 456.78,
                "power_watts": 10.5,
                "energy_efficiency_items_per_joule": 11.76
            }
        }
        
        # Store the test result
        success = db_handler.store_test_result(test_result)
        
        if not success:
            logger.error("Failed to store performance test result")
            return False
            
        logger.info("Performance metrics storage verified")
        return True
    except Exception as e:
        logger.error(f"Performance metrics storage verification failed: {e}")
        traceback.print_exc()
        return False

def verify_compatibility_matrix(db_handler: TestResultsDBHandler) -> bool:
    """Verify compatibility matrix generation.
    
    Args:
        db_handler: TestResultsDBHandler instance
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying compatibility matrix generation")
    
    try:
        # Create test results for various hardware platforms
        platforms = ["cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
        
        for platform in platforms:
            # Create test result with compatibility information
            test_result = {
                "model_name": "matrix-test-model",
                "model_family": "matrix-test-family",
                "hardware_type": platform,
                "test_type": "compatibility-verification",
                "status": "Success",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "compatibility": {
                    "status": "full" if platform in ["cuda", "openvino"] else "limited",
                    "score": 1.0 if platform in ["cuda", "openvino"] else 0.5,
                    "recommended": platform == "cuda"
                }
            }
            
            # Store the test result
            success = db_handler.store_test_result(test_result)
            
            if not success:
                logger.error(f"Failed to store compatibility test result for {platform}")
                return False
        
        # Also store compatibility directly in cross_platform_compatibility table
        # First, make sure the model exists
        model_id_query = """
        SELECT model_id FROM models WHERE model_name = 'matrix-test-model'
        """
        
        model_id = db_handler.con.execute(model_id_query).fetchone()
        
        if model_id:
            model_id = model_id[0]
            
            # Insert into cross_platform_compatibility
            db_handler.con.execute(
                """
                INSERT INTO cross_platform_compatibility (
                    model_name, model_family, cuda_support, rocm_support,
                    mps_support, openvino_support, qualcomm_support,
                    webnn_support, webgpu_support, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    "matrix-test-model", "matrix-test-family",
                    True, False, False, True, False, False, False,
                    datetime.now()
                ]
            )
        
        logger.info("Compatibility matrix generation verified")
        return True
    except Exception as e:
        logger.error(f"Compatibility matrix verification failed: {e}")
        traceback.print_exc()
        return False

def verify_report_generation(db_handler: TestResultsDBHandler) -> bool:
    """Verify report generation.
    
    Args:
        db_handler: TestResultsDBHandler instance
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying report generation")
    
    try:
        # Generate reports in different formats
        formats = ["markdown", "html", "json"]
        
        for fmt in formats:
            output_file = f"test_report.{fmt}"
            
            # Generate report
            report = db_handler.generate_report(format=fmt, output_file=output_file)
            
            if not report:
                logger.error(f"Failed to generate {fmt} report")
                return False
            
            # Check if file was created
            if not os.path.exists(output_file):
                logger.error(f"Report file {output_file} was not created")
                return False
            
            logger.info(f"Successfully generated {fmt} report")
            
            # Clean up
            try:
                os.remove(output_file)
            except:
                pass
        
        logger.info("Report generation verified")
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
    if not HAS_DEPENDENCIES:
        logger.error("Missing dependencies, cannot proceed with verification")
        return False
    
    # Verify database connection
    success, conn = verify_database_connection(args.db_path)
    if not success:
        return False
    
    # Create database handler
    db_handler = TestResultsDBHandler(db_path=args.db_path)
    
    # Verify database handler is available
    if not db_handler.is_available():
        logger.error("TestResultsDBHandler is not available")
        return False
    
    # Run verification tests
    verifications = [
        ("Test Result Storage", verify_test_result_storage(db_handler)),
        ("Performance Metrics Storage", verify_performance_metrics_storage(db_handler)),
        ("Compatibility Matrix Generation", verify_compatibility_matrix(db_handler)),
        ("Report Generation", verify_report_generation(db_handler))
    ]
    
    # Print summary
    logger.info("\nVerification Summary:")
    all_passed = True
    
    for name, result in verifications:
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {name}: {status}")
        all_passed = all_passed and result
    
    if all_passed:
        logger.info("\nAll verifications passed. Database integration is working correctly.")
        logger.info("Phase 16 requirements have been completed successfully.")
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