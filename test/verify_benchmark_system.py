#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark System Verification 

This script verifies that the benchmark system properly handles hardware
availability and simulation status. It helps ensure that:

1. Hardware detection is properly working
2. Hardware simulation flags are correctly set in benchmark results
3. Database schema includes simulation flags
4. Simulation status is reflected in reports

These checks help fulfill the requirements from NEXT_STEPS.md item #10:
"Critical Benchmark System Issues"
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_hardware_availability() -> Dict[str, bool]:
    """
    Verify hardware availability detection with the fixed hardware detection system.
    
    Returns:
        Dictionary mapping hardware types to their availability status
    """
    try:
        # Import the fixed hardware detection
        from hardware_detection import (
            detect_available_hardware,
            CPU, CUDA, ROCM, MPS, OPENVINO, WEBNN, WEBGPU, QUALCOMM
        )
        
        # Run detection
        results = detect_available_hardware()
        
        # Extract real hardware (not simulated)
        real_hardware = results.get("real_hardware", {})
        simulated_hardware = results.get("simulated_hardware", {})
        
        # Log results
        logger.info("Hardware detection results:")
        
        if real_hardware:
            real_hw_list = [hw for hw, is_real in real_hardware.items() if is_real]
            logger.info(f"Real hardware detected: {', '.join(real_hw_list)}")
        
        if simulated_hardware:
            sim_hw_list = [hw for hw, is_sim in simulated_hardware.items() if is_sim]
            logger.warning(f"Simulated hardware detected: {', '.join(sim_hw_list)}")
            
        # Prepare a clean result without internal details
        hardware_status = {
            CPU: True,  # CPU is always available
            CUDA: results["hardware"].get(CUDA, False) and real_hardware.get(CUDA, False),
            ROCM: results["hardware"].get(ROCM, False) and real_hardware.get(ROCM, False),
            MPS: results["hardware"].get(MPS, False) and real_hardware.get(MPS, False),
            OPENVINO: results["hardware"].get(OPENVINO, False) and real_hardware.get(OPENVINO, False),
            QUALCOMM: results["hardware"].get(QUALCOMM, False) and real_hardware.get(QUALCOMM, False),
            WEBNN: results["hardware"].get(WEBNN, False) and real_hardware.get(WEBNN, False),
            WEBGPU: results["hardware"].get(WEBGPU, False) and real_hardware.get(WEBGPU, False)
        }
        
        return hardware_status
        
    except ImportError:
        logger.error("Could not import hardware_detection module. Make sure the fixes have been applied.")
        return {
            "cpu": True,
            "cuda": False,
            "rocm": False,
            "mps": False,
            "openvino": False,
            "qualcomm": False,
            "webnn": False,
            "webgpu": False
        }
    except Exception as e:
        logger.error(f"Error during hardware verification: {str(e)}")
        traceback.print_exc()
        return {
            "cpu": True,
            "cuda": False,
            "rocm": False,
            "mps": False,
            "openvino": False,
            "qualcomm": False,
            "webnn": False,
            "webgpu": False
        }

def verify_benchmark_results(db_path: str) -> Dict[str, Any]:
    """
    Verify benchmark results in the database to ensure simulation status is correctly stored.
    
    Args:
        db_path: Path to the DuckDB database
        
    Returns:
        Dictionary with verification results
    """
    results = {
        "database_schema_check": False,
        "simulation_flags_present": False,
        "real_vs_simulated_counts": {},
        "errors": []
    }
    
    try:
        # Try to import DuckDB
        import duckdb
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Check if performance_results table exists
        table_exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'performance_results'"
        ).fetchone()[0] > 0
        
        if not table_exists:
            results["errors"].append("performance_results table not found in database")
            return results
            
        # Check if simulation flags are present in schema
        try:
            schema_check = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'performance_results' AND column_name IN ('is_simulated', 'simulation_reason')"
            ).fetchall()
            
            simulation_columns = [row[0] for row in schema_check]
            results["simulation_flags_present"] = len(simulation_columns) > 0
            results["database_schema_check"] = "is_simulated" in simulation_columns
            
            if not results["simulation_flags_present"]:
                results["errors"].append("Simulation flags not found in database schema")
            
            # Count real vs simulated results
            if "is_simulated" in simulation_columns:
                # Get counts
                counts = conn.execute(
                    "SELECT is_simulated, COUNT(*) FROM performance_results GROUP BY is_simulated"
                ).fetchall()
                
                for is_simulated, count in counts:
                    if is_simulated:
                        results["real_vs_simulated_counts"]["simulated"] = count
                    else:
                        results["real_vs_simulated_counts"]["real"] = count
                        
                # Make sure all entries have simulation status
                null_count = conn.execute(
                    "SELECT COUNT(*) FROM performance_results WHERE is_simulated IS NULL"
                ).fetchone()[0]
                
                if null_count > 0:
                    results["errors"].append(f"{null_count} benchmark entries have NULL simulation status")
                    results["real_vs_simulated_counts"]["unknown"] = null_count
            
        except Exception as e:
            results["errors"].append(f"Error checking schema: {str(e)}")
            
        # Close connection
        conn.close()
        
    except ImportError:
        results["errors"].append("DuckDB not available, could not check database")
    except Exception as e:
        results["errors"].append(f"Unexpected error: {str(e)}")
        traceback.print_exc()
    
    return results

def verify_benchmark_reports() -> Dict[str, bool]:
    """
    Verify benchmark reports to ensure they indicate simulation status.
    
    Returns:
        Dictionary with verification results
    """
    results = {
        "simulation_warnings_in_reports": False,
        "simulation_indicators_in_matrix": False
    }
    
    # Check for report files
    report_files = list(Path(".").glob("**/model_hardware_report_*.md"))
    
    if not report_files:
        logger.warning("No benchmark report files found")
        return results
    
    # Check the most recent report
    latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_report, "r") as f:
            report_content = f.read()
        
        # Check for simulation warnings
        results["simulation_warnings_in_reports"] = "SIMULATED" in report_content or "⚠️" in report_content
        
        # Check for simulation indicators in compatibility matrix
        results["simulation_indicators_in_matrix"] = "⚠️" in report_content and "hardware platforms are being simulated" in report_content
        
    except Exception as e:
        logger.error(f"Error checking report file {latest_report}: {str(e)}")
    
    return results

def verify_json_outputs() -> Dict[str, bool]:
    """
    Verify JSON benchmark outputs to ensure they include simulation status.
    
    Returns:
        Dictionary with verification results
    """
    results = {
        "simulation_flags_in_json": False
    }
    
    # Check for benchmark result files
    json_files = list(Path(".").glob("**/benchmark_results/*.json"))
    
    if not json_files:
        logger.warning("No benchmark JSON files found")
        return results
    
    # Check the most recent JSON file
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_json, "r") as f:
            json_data = json.load(f)
        
        # Check for simulation flags
        if isinstance(json_data, dict):
            results["simulation_flags_in_json"] = (
                "is_simulated" in json_data or 
                "simulation_mode" in json_data or 
                "simulated" in json_data
            )
            
    except Exception as e:
        logger.error(f"Error checking JSON file {latest_json}: {str(e)}")
    
    return results

def main():
    """Main function for running benchmark system verification"""
    parser = argparse.ArgumentParser(description="Benchmark System Verification")
    
    parser.add_argument("--db-path", type=str, default=None, 
                      help="Path to DuckDB database (defaults to environment variable BENCHMARK_DB_PATH)")
    parser.add_argument("--hardware-only", action="store_true", 
                      help="Only verify hardware detection")
    parser.add_argument("--schema-only", action="store_true", 
                      help="Only verify database schema")
    parser.add_argument("--reports-only", action="store_true", 
                      help="Only verify benchmark reports")
    parser.add_argument("--json-only", action="store_true", 
                      help="Only verify JSON outputs")
    parser.add_argument("--apply-fixes", action="store_true", 
                      help="Apply fixes to database schema if needed")
    parser.add_argument("--verbose", action="store_true", 
                      help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get database path
    db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Header
    print("\n" + "=" * 80)
    print("BENCHMARK SYSTEM VERIFICATION")
    print("=" * 80)
    
    # Determine which verifications to run
    run_all = not (args.hardware_only or args.schema_only or args.reports_only or args.json_only)
    
    # Track overall status
    all_checks_passed = True
    
    # Verify hardware detection
    if run_all or args.hardware_only:
        print("\n1. Hardware Detection Verification")
        print("-" * 40)
        
        hardware_status = verify_hardware_availability()
        
        print("\nHardware Availability:")
        for hw, available in hardware_status.items():
            status = "✅ AVAILABLE" if available else "❌ NOT AVAILABLE"
            print(f"  {hw}: {status}")
            
        hardware_check_passed = True
        
        # Count available hardware
        available_count = sum(1 for hw, available in hardware_status.items() if available)
        if available_count <= 1:  # Only CPU
            logger.warning("⚠️  Only CPU hardware detected, recommend checking hardware setup")
            hardware_check_passed = False
            
        if hardware_check_passed:
            print("\n✅ Hardware detection check PASSED")
        else:
            print("\n⚠️  Hardware detection check has WARNINGS")
            all_checks_passed = False
    
    # Verify database schema
    if run_all or args.schema_only:
        print("\n2. Database Schema Verification")
        print("-" * 40)
        
        if os.path.exists(db_path):
            schema_results = verify_benchmark_results(db_path)
            
            print(f"\nDatabase: {db_path}")
            print(f"  Schema includes simulation flags: {'✅ YES' if schema_results['database_schema_check'] else '❌ NO'}")
            
            if schema_results['real_vs_simulated_counts']:
                print("\nBenchmark Result Counts:")
                for result_type, count in schema_results['real_vs_simulated_counts'].items():
                    print(f"  {result_type}: {count}")
            
            if schema_results['errors']:
                print("\nErrors:")
                for error in schema_results['errors']:
                    print(f"  ❌ {error}")
                    
                schema_check_passed = False
            else:
                schema_check_passed = schema_results['database_schema_check']
                
            if schema_check_passed:
                print("\n✅ Database schema check PASSED")
            else:
                print("\n❌ Database schema check FAILED")
                all_checks_passed = False
                
                # Offer to apply schema fixes
                if args.apply_fixes and not schema_results['database_schema_check']:
                    print("\nAttempting to apply schema fixes...")
                    try:
                        import duckdb
                        conn = duckdb.connect(db_path)
                        
                        # Add simulation columns if they don't exist
                        columns_to_add = []
                        if "is_simulated" not in schema_results.get("simulation_columns", []):
                            columns_to_add.append("is_simulated BOOLEAN DEFAULT FALSE")
                        if "simulation_reason" not in schema_results.get("simulation_columns", []):
                            columns_to_add.append("simulation_reason VARCHAR")
                            
                        if columns_to_add:
                            for column_def in columns_to_add:
                                try:
                                    conn.execute(f"ALTER TABLE performance_results ADD COLUMN {column_def}")
                                    print(f"✅ Added column: {column_def}")
                                except Exception as e:
                                    print(f"❌ Error adding column: {str(e)}")
                                    
                        conn.close()
                        print("\nSchema fixes applied. Please run verification again to confirm.")
                    except Exception as e:
                        print(f"❌ Error applying schema fixes: {str(e)}")
        else:
            print(f"\n❌ Database not found at: {db_path}")
            all_checks_passed = False
    
    # Verify benchmark reports
    if run_all or args.reports_only:
        print("\n3. Benchmark Reports Verification")
        print("-" * 40)
        
        report_results = verify_benchmark_reports()
        
        print("\nReport Checks:")
        print(f"  Simulation warnings in reports: {'✅ YES' if report_results['simulation_warnings_in_reports'] else '❌ NO'}")
        print(f"  Simulation indicators in matrix: {'✅ YES' if report_results['simulation_indicators_in_matrix'] else '❌ NO'}")
        
        reports_check_passed = report_results['simulation_warnings_in_reports'] and report_results['simulation_indicators_in_matrix']
        
        if reports_check_passed:
            print("\n✅ Benchmark reports check PASSED")
        else:
            print("\n⚠️  Benchmark reports check has WARNINGS")
            print("    Reports should be regenerated after fixes are applied")
            all_checks_passed = False
    
    # Verify JSON outputs
    if run_all or args.json_only:
        print("\n4. JSON Outputs Verification")
        print("-" * 40)
        
        json_results = verify_json_outputs()
        
        print("\nJSON Checks:")
        print(f"  Simulation flags in JSON files: {'✅ YES' if json_results['simulation_flags_in_json'] else '❌ NO'}")
        
        json_check_passed = json_results['simulation_flags_in_json']
        
        if json_check_passed:
            print("\n✅ JSON outputs check PASSED")
        else:
            print("\n⚠️  JSON outputs check has WARNINGS")
            print("    Note: This is expected if JSON output is deprecated in favor of database storage")
    
    # Final summary
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✅ ALL BENCHMARK SYSTEM CHECKS PASSED")
    else:
        print("⚠️  SOME BENCHMARK SYSTEM CHECKS FAILED OR HAVE WARNINGS")
        print("   See details above for specific issues and recommendations")
    print("=" * 80 + "\n")
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())