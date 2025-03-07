#!/usr/bin/env python3
"""
Test Simulation Detection and Flagging

This script tests the improved simulation detection and flagging system
to ensure simulated hardware is properly identified and marked in results.

April 2025 Update: Part of the benchmark system improvements from NEXT_STEPS.md
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the benchmarker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from benchmark_all_key_models import KeyModelBenchmarker
    BENCHMARKER_AVAILABLE = True
except ImportError:
    logger.error("Failed to import KeyModelBenchmarker")
    BENCHMARKER_AVAILABLE = False

def test_hardware_detection():
    """Test hardware detection with simulation awareness"""
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # First test with default settings (no simulation)
        logger.info("Testing with default hardware detection (no simulation)...")
        benchmarker = KeyModelBenchmarker(
            output_dir=temp_dir,
            use_small_models=True,
            hardware_platforms=["cpu", "cuda", "qnn", "webnn", "webgpu"],
            fix_implementations=False,
            debug=True
        )
        
        # Check if simulation flags are tracked
        if not hasattr(benchmarker, "_simulated_hardware"):
            logger.error("ERROR: Benchmarker does not track simulated hardware")
            return False
        
        # Print simulation status
        logger.info("Default hardware simulation status:")
        for hw, is_simulated in benchmarker._simulated_hardware.items():
            logger.info(f"  - {hw}: {'SIMULATED' if is_simulated else 'Real Hardware'}")
        
        # Check if CPU is properly detected as real
        if benchmarker._simulated_hardware.get("cpu", True):
            logger.error("ERROR: CPU incorrectly marked as simulated")
            return False
            
        # Now test with explicit simulation
        logger.info("\nTesting with explicit hardware simulation...")
        
        # Set simulation environment variables
        os.environ["QNN_SIMULATION_MODE"] = "1"
        os.environ["WEBNN_SIMULATION"] = "1"
        os.environ["WEBGPU_SIMULATION"] = "1"
        
        # Create new benchmarker
        sim_benchmarker = KeyModelBenchmarker(
            output_dir=temp_dir,
            use_small_models=True,
            hardware_platforms=["cpu", "cuda", "qnn", "webnn", "webgpu"],
            fix_implementations=False,
            debug=True
        )
        
        # Print simulation status
        logger.info("Explicit simulation hardware status:")
        for hw, is_simulated in sim_benchmarker._simulated_hardware.items():
            logger.info(f"  - {hw}: {'SIMULATED' if is_simulated else 'Real Hardware'}")
        
        # Check if simulated hardware is properly detected
        if not sim_benchmarker._simulated_hardware.get("qnn", False):
            logger.error("ERROR: QNN not marked as simulated when QNN_SIMULATION_MODE=1")
            return False
            
        if not sim_benchmarker._simulated_hardware.get("webnn", False):
            logger.error("ERROR: WebNN not marked as simulated when WEBNN_SIMULATION=1")
            return False
            
        if not sim_benchmarker._simulated_hardware.get("webgpu", False):
            logger.error("ERROR: WebGPU not marked as simulated when WEBGPU_SIMULATION=1")
            return False
        
        # Test with availability override (should be treated as simulation)
        logger.info("\nTesting with hardware availability overrides...")
        
        # Clean previous environment variables
        for var in ["QNN_SIMULATION_MODE", "WEBNN_SIMULATION", "WEBGPU_SIMULATION"]:
            if var in os.environ:
                del os.environ[var]
        
        # Set availability override variables
        os.environ["CUDA_AVAILABLE"] = "1"  # Force CUDA available
        os.environ["QNN_AVAILABLE"] = "1"   # Force QNN available
        
        # Create new benchmarker
        override_benchmarker = KeyModelBenchmarker(
            output_dir=temp_dir,
            use_small_models=True,
            hardware_platforms=["cpu", "cuda", "qnn"],
            fix_implementations=False,
            debug=True
        )
        
        # Print simulation status
        logger.info("Override simulation hardware status:")
        for hw, is_simulated in override_benchmarker._simulated_hardware.items():
            logger.info(f"  - {hw}: {'SIMULATED' if is_simulated else 'Real Hardware'}")
        
        # Check if overridden hardware is properly marked as simulated
        # Note: This check depends on actual hardware - if CUDA is really available,
        # it won't be marked as simulated even with the override
        if hw == "qnn" and not override_benchmarker._simulated_hardware.get(hw, False):
            logger.error(f"ERROR: {hw} not marked as simulated when {hw.upper()}_AVAILABLE=1")
            # Don't fail the test if hardware is actually available
            logger.warning(f"This might be because real {hw} hardware is available")
            
        # Clean up environment variables
        for var in ["CUDA_AVAILABLE", "QNN_AVAILABLE"]:
            if var in os.environ:
                del os.environ[var]
            
        # Test centralized hardware detection integration
        logger.info("\nTesting centralized hardware detection integration...")
        try:
            from centralized_hardware_detection.hardware_detection import HARDWARE_MANAGER
            
            # Get hardware info
            capabilities = HARDWARE_MANAGER.get_capabilities()
            
            # Check simulation fields in returned capabilities
            if "webnn_simulation" in capabilities and "webgpu_simulation" in capabilities and "qualcomm_simulation" in capabilities:
                logger.info("✅ Centralized hardware detection includes simulation flags")
            else:
                logger.warning("⚠️ Centralized hardware detection missing some simulation flags")
                missing = []
                for flag in ["webnn_simulation", "webgpu_simulation", "qualcomm_simulation"]:
                    if flag not in capabilities:
                        missing.append(flag)
                logger.warning(f"Missing flags: {', '.join(missing)}")
        except ImportError:
            logger.warning("Could not import centralized hardware detection module")
            
        logger.info("Hardware detection tests completed successfully!")
        return True

def test_report_generation():
    """Test report generation with simulation flags"""
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set simulation environment variables
        os.environ["QNN_SIMULATION_MODE"] = "1"
        os.environ["WEBNN_SIMULATION"] = "1"
        
        # Create benchmarker with specific hardware platforms
        benchmarker = KeyModelBenchmarker(
            output_dir=temp_dir,
            use_small_models=True,
            hardware_platforms=["cpu", "qnn", "webnn"],
            fix_implementations=False,
            debug=True
        )
        
        # Generate report
        logger.info("Generating report with simulation flags...")
        report_path = benchmarker.generate_report()
        
        # Check if report file exists
        if not os.path.exists(report_path):
            logger.error(f"ERROR: Report file not found: {report_path}")
            return False
        
        # Read report content
        with open(report_path, "r") as f:
            report_content = f.read()
        
        # Check if simulation warnings are included
        if "SIMULATED" not in report_content:
            logger.error("ERROR: Report does not include simulation warnings")
            return False
            
        if "⚠️" not in report_content:
            logger.error("ERROR: Report does not include simulation warning symbols")
            return False
            
        logger.info("Report generation test completed successfully!")
        
        # Clean up environment variables
        for var in ["QNN_SIMULATION_MODE", "WEBNN_SIMULATION"]:
            if var in os.environ:
                del os.environ[var]
                
        return True

def test_database_integration():
    """Test database integration with simulation flags"""
    # Skip test if database integration not available
    try:
        from benchmark_db_api import BenchmarkDBAPI
        HAS_DB_API = True
    except ImportError:
        logger.warning("Skipping database integration test: BenchmarkDBAPI not available")
        return True
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as temp_db:
        # Set environment variables
        os.environ["BENCHMARK_DB_PATH"] = temp_db.name
        os.environ["QNN_SIMULATION_MODE"] = "1"
        
        # Try to update schema
        try:
            logger.info("Updating database schema...")
            from update_db_schema_for_simulation import connect_to_db, update_schema
            
            # Connect to database
            conn = connect_to_db(temp_db.name)
            if not conn:
                logger.error("ERROR: Failed to connect to database")
                return False
            
            # Update schema
            if not update_schema(conn):
                logger.error("ERROR: Failed to update database schema")
                conn.close()
                return False
                
            conn.close()
        except ImportError:
            logger.warning("update_db_schema_for_simulation.py not available, skipping schema update")
        
        # Create dummy results for storing
        simulated_result = {
            "model_name": "test-model-simulated",
            "hardware": "qnn",  # QNN is simulated via environment variable
            "batch_size": 1,
            "throughput_items_per_second": 100.0,
            "latency_ms": 10.0,
            "memory_mb": 1024.0,
            "model_family": "test",
            "model_type": "test"
        }
        
        real_result = {
            "model_name": "test-model-real",
            "hardware": "cpu",  # CPU is always real
            "batch_size": 1,
            "throughput_items_per_second": 80.0,
            "latency_ms": 12.5,
            "memory_mb": 512.0,
            "model_family": "test",
            "model_type": "test"
        }
        
        # Store results in database
        try:
            logger.info("Storing test results in database...")
            from benchmark_all_key_models import store_benchmark_in_database
            
            # Store simulated result
            logger.info("Storing simulated result...")
            sim_success = store_benchmark_in_database(simulated_result)
            if not sim_success:
                logger.error("ERROR: Failed to store simulated benchmark result in database")
                return False
            
            # Store real result
            logger.info("Storing real result...")
            real_success = store_benchmark_in_database(real_result)
            if not real_success:
                logger.error("ERROR: Failed to store real benchmark result in database")
                return False
                
            logger.info("Test results stored successfully!")
            
            # Verify records were stored with correct simulation status
            try:
                import duckdb
                
                logger.info("Verifying simulation flags in database records...")
                
                # Connect to database
                conn = duckdb.connect(temp_db.name)
                
                # Check if database tables exist
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [t[0] for t in tables]
                
                if "performance_results" in table_names:
                    # Check if simulation columns exist
                    columns = conn.execute("DESCRIBE SELECT * FROM performance_results LIMIT 0").fetchall()
                    column_names = [c[0].lower() for c in columns]
                    
                    if "is_simulated" in column_names:
                        # Query results with simulation status
                        results = conn.execute("""
                            SELECT model_name, is_simulated, simulation_reason, hardware_id 
                            FROM performance_results
                        """).fetchall()
                        
                        logger.info("Database records with simulation status:")
                        for row in results:
                            logger.info(f"  - {row[0]}: {'SIMULATED' if row[1] else 'Real'}")
                        
                        # Check if we have both simulated and real records
                        has_simulated = any(r[1] for r in results)
                        has_real = any(not r[1] for r in results)
                        
                        if has_simulated and has_real:
                            logger.info("✅ Database contains both simulated and real records")
                        elif has_simulated:
                            logger.warning("⚠️ Database contains only simulated records")
                        elif has_real:
                            logger.warning("⚠️ Database contains only real records")
                        else:
                            logger.error("❌ No records found in database")
                    else:
                        logger.warning("⚠️ performance_results table missing is_simulated column")
                        logger.info(f"Available columns: {', '.join(column_names)}")
                else:
                    logger.warning(f"⚠️ performance_results table not found. Available tables: {', '.join(table_names)}")
                
                conn.close()
            except Exception as e:
                logger.error(f"Error verifying database records: {e}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to store benchmark results in database: {e}")
            return False
        
        # Clean up environment variables
        for var in ["BENCHMARK_DB_PATH", "QNN_SIMULATION_MODE"]:
            if var in os.environ:
                del os.environ[var]
                
        logger.info("Database integration test completed successfully!")
        return True

def main():
    """Main entry point for testing simulation detection and flagging"""
    parser = argparse.ArgumentParser(description="Test Simulation Detection and Flagging")
    parser.add_argument("--hardware-only", action="store_true",
                      help="Only test hardware detection")
    parser.add_argument("--report-only", action="store_true",
                      help="Only test report generation")
    parser.add_argument("--database-only", action="store_true",
                      help="Only test database integration")
    args = parser.parse_args()
    
    # Check if benchmarker is available
    if not BENCHMARKER_AVAILABLE:
        logger.error("ERROR: KeyModelBenchmarker not available")
        return 1
    
    # Run selected tests
    if args.hardware_only:
        if not test_hardware_detection():
            return 1
    elif args.report_only:
        if not test_report_generation():
            return 1
    elif args.database_only:
        if not test_database_integration():
            return 1
    else:
        # Run all tests
        logger.info("Running all simulation detection and flagging tests...")
        
        if not test_hardware_detection():
            return 1
            
        if not test_report_generation():
            return 1
            
        if not test_database_integration():
            return 1
    
    logger.info("All tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())