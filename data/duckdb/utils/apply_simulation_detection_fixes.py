#!/usr/bin/env python
"""
Apply simulation detection fixes to hardware detection and database.

This script addresses critical benchmark system issues by:
1. Updating database schema to add simulation tracking columns
2. Creating hardware availability logging table
3. Flagging existing simulated results in the database
4. Installing enhanced hardware detection module

Implementation date: April 8, 2025
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
import shutil

try:
    import duckdb
except ImportError:
    print("Error: duckdb module not found. Please install it with pip install duckdb.")
    sys.exit(1)

from hardware_detection_updates import get_simulation_tracking_schema_updates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection
def get_db_connection(db_path=None):
    """Get a connection to the benchmark database"""
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    logger.info(f"Connecting to database: {db_path}")
    return duckdb.connect(db_path)

def update_database_schema(conn, backup=True):
    """Apply schema updates to add simulation tracking columns"""
    try:
        # Create a backup of the database first if requested
        if backup:
            db_path = conn.execute("PRAGMA database_list").fetchone()[2]
            backup_path = f"{db_path}.bak_simulation_fixes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Creating database backup: {backup_path}")
            shutil.copy2(db_path, backup_path)
        
        # Get schema updates from the updates module
        schema_updates = get_simulation_tracking_schema_updates()
        
        # Apply each schema update
        for table_name, sql in schema_updates.items():
            logger.info(f"Applying schema update for {table_name}")
            try:
                conn.execute(sql)
                conn.commit()
                logger.info(f"Successfully updated schema for {table_name}")
            except Exception as e:
                logger.error(f"Error updating schema for {table_name}: {str(e)}")
                raise
        
        return True
    except Exception as e:
        logger.error(f"Database schema update failed: {str(e)}")
        return False

def create_hardware_availability_log(conn):
    """Create the hardware availability log table if it doesn't exist"""
    try:
        logger.info("Creating hardware availability log table if not exists")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_availability_log (
            id INTEGER PRIMARY KEY,
            hardware_type VARCHAR,
            is_available BOOLEAN,
            is_simulated BOOLEAN DEFAULT FALSE,
            detection_method VARCHAR,
            detection_details JSON,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error creating hardware availability log table: {str(e)}")
        return False

def flag_simulated_results(conn):
    """Flag existing results that are likely from simulated hardware"""
    try:
        # Get hardware_id for potentially simulated hardware types
        logger.info("Finding hardware IDs for potentially simulated hardware")
        hardware_ids = conn.execute("""
        SELECT hardware_id, hardware_type FROM hardware_platforms
        WHERE hardware_type IN ('webnn', 'webgpu', 'qualcomm')
        """).fetchall()
        
        if not hardware_ids:
            logger.info("No potentially simulated hardware found in database")
            return True
        
        # For each hardware type, flag existing results
        total_flagged = 0
        for hardware_id, hardware_type in hardware_ids:
            logger.info(f"Checking results for hardware_type={hardware_type}, hardware_id={hardware_id}")
            
            # Create simulation reason by hardware type
            if hardware_type == "webnn":
                reason = "WebNN in desktop environment is likely simulated"
            elif hardware_type == "webgpu":
                reason = "WebGPU in desktop environment is likely simulated"
            elif hardware_type == "qualcomm":
                reason = "Qualcomm hardware not physically present, results are simulated"
            else:
                reason = f"Hardware type {hardware_type} is likely simulated based on environment"
            
            # Update test_results
            flagged_tests = conn.execute("""
            UPDATE test_results
            SET 
                is_simulated = TRUE,
                simulation_reason = ?,
                error_category = CASE 
                    WHEN success = FALSE AND error_category IS NULL THEN 'hardware_not_available' 
                    ELSE error_category 
                END,
                error_details = CASE 
                    WHEN success = FALSE AND error_details IS NULL THEN json('{"simulation_detected": true}')
                    ELSE error_details 
                END
            WHERE
                hardware_id = ?
                AND (is_simulated IS NULL OR is_simulated = FALSE)
            RETURNING id
            """, [reason, hardware_id]).fetchall()
            
            # Update performance_results
            flagged_performance = conn.execute("""
            UPDATE performance_results
            SET 
                is_simulated = TRUE,
                simulation_reason = ?
            WHERE
                hardware_id = ?
                AND (is_simulated IS NULL OR is_simulated = FALSE)
            RETURNING id
            """, [reason, hardware_id]).fetchall()
            
            n_tests = len(flagged_tests) if flagged_tests else 0
            n_perf = len(flagged_performance) if flagged_performance else 0
            total_flagged += n_tests + n_perf
            
            logger.info(f"Flagged {n_tests} test results and {n_perf} performance results for {hardware_type}")
        
        # Check for environment variable override patterns in error messages
        logger.info("Checking for environment variable override patterns in error messages...")
        
        # Find test results with environment variable override patterns
        patterns = [
            "%WEBNN_SIMULATION%", 
            "%WEBGPU_SIMULATION%", 
            "%QNN_SIMULATION%",
            "%WEBNN_AVAILABLE%",
            "%WEBGPU_AVAILABLE%"
        ]
        
        # For each pattern, update matching results
        pattern_flagged = 0
        for pattern in patterns:
            hw_type = None
            if "WEBNN" in pattern:
                hw_type = "webnn"
            elif "WEBGPU" in pattern:
                hw_type = "webgpu"
            elif "QNN" in pattern:
                hw_type = "qualcomm"
                
            if not hw_type:
                continue
                
            # Get hardware ID for this type
            hw_id_result = conn.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
                [hw_type]
            ).fetchone()
            
            if not hw_id_result:
                continue
                
            hw_id = hw_id_result[0]
            reason = f"Detected {hw_type} environment variable override in error message"
            
            # Update matching results
            flagged = conn.execute("""
            UPDATE test_results
            SET 
                is_simulated = TRUE,
                simulation_reason = ?
            WHERE
                (error_message LIKE ? OR details LIKE ?)
                AND (is_simulated IS NULL OR is_simulated = FALSE)
            RETURNING id
            """, [reason, pattern, pattern]).fetchall()
            
            n_flagged = len(flagged) if flagged else 0
            pattern_flagged += n_flagged
            
            if n_flagged > 0:
                logger.info(f"Flagged {n_flagged} test results with pattern '{pattern}'")
        
        # Total flagged count
        total_flagged += pattern_flagged
        
        conn.commit()
        logger.info(f"Total results flagged as simulated: {total_flagged}")
        return True
    except Exception as e:
        logger.error(f"Error flagging simulated results: {str(e)}")
        conn.rollback()
        return False

def add_hardware_detection_log_entry(conn, hardware_info):
    """Add an entry to the hardware availability log"""
    try:
        # Parse the information from hardware detection
        now = datetime.datetime.now()
        
        # Add an entry for each hardware type
        for hardware_type, is_available in hardware_info.get("hardware", {}).items():
            # Check if this hardware is simulated
            is_simulated = hardware_type in hardware_info.get("simulated_hardware", [])
            
            # Get detection details for this hardware type
            detection_details = hardware_info.get("details", {}).get(hardware_type, {})
            detection_method = "enhanced_detection"
            
            # Insert the log entry
            conn.execute("""
            INSERT INTO hardware_availability_log (
                hardware_type, is_available, is_simulated, 
                detection_method, detection_details, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, [
                hardware_type, 
                is_available, 
                is_simulated, 
                detection_method, 
                json.dumps(detection_details), 
                now
            ])
        
        conn.commit()
        logger.info(f"Added hardware detection log entries for {len(hardware_info.get('hardware', {}))} hardware types")
        return True
    except Exception as e:
        logger.error(f"Error adding hardware detection log entry: {str(e)}")
        conn.rollback()
        return False

def run_hardware_detection_test():
    """Run the enhanced hardware detection and return the results"""
    try:
        from hardware_detection_updates import detect_hardware_with_simulation_check
        logger.info("Running enhanced hardware detection")
        hardware_info = detect_hardware_with_simulation_check()
        return hardware_info
    except Exception as e:
        logger.error(f"Error running enhanced hardware detection: {str(e)}")
        return None

def main():
    """Main function to apply simulation detection fixes"""
    parser = argparse.ArgumentParser(description="Apply simulation detection fixes")
    parser.add_argument("--db-path", help="Path to the benchmark database")
    parser.add_argument("--no-backup", action="store_true", help="Skip database backup")
    parser.add_argument("--test-only", action="store_true", help="Only test detection without modifying database")
    args = parser.parse_args()
    
    logger.info("Starting simulation detection fixes application")
    
    # Run hardware detection test to validate the enhanced detection
    hardware_info = run_hardware_detection_test()
    if not hardware_info:
        logger.error("Hardware detection test failed. Aborting.")
        return False
    
    # Print simulation status
    logger.info("Hardware simulation status:")
    for hw_type in hardware_info.get("simulated_hardware", []):
        logger.info(f"  {hw_type}: SIMULATED")
    
    if args.test_only:
        logger.info("Test-only mode - no database changes will be made")
        return True
    
    # Connect to the database
    try:
        conn = get_db_connection(args.db_path)
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return False
    
    # Apply database schema updates
    success = update_database_schema(conn, not args.no_backup)
    if not success:
        logger.error("Failed to update database schema. Aborting.")
        return False
    
    # Create hardware availability log
    success = create_hardware_availability_log(conn)
    if not success:
        logger.error("Failed to create hardware availability log. Aborting.")
        return False
    
    # Flag existing simulated results
    success = flag_simulated_results(conn)
    if not success:
        logger.error("Failed to flag simulated results. Aborting.")
        return False
    
    # Add current hardware detection log entry
    success = add_hardware_detection_log_entry(conn, hardware_info)
    if not success:
        logger.error("Failed to add hardware detection log entry. Aborting.")
        return False
    
    logger.info("Successfully applied simulation detection fixes")
    
    # Final validation query to check simulation tracking
    try:
        # Check if simulation columns exist and have values
        sim_check = conn.execute("""
        SELECT 
            COUNT(*) as total_results,
            COUNT(CASE WHEN is_simulated IS NOT NULL THEN 1 END) as results_with_sim_flag,
            COUNT(CASE WHEN is_simulated = TRUE THEN 1 END) as simulated_results
        FROM test_results
        """).fetchone()
        
        if sim_check:
            total, with_flag, simulated = sim_check
            logger.info(f"Database validation: {total} total results, {with_flag} with simulation flag, {simulated} flagged as simulated")
        
        # Check hardware availability log
        log_check = conn.execute("""
        SELECT COUNT(*) FROM hardware_availability_log
        """).fetchone()
        
        if log_check:
            logger.info(f"Hardware availability log has {log_check[0]} entries")
    except Exception as e:
        logger.error(f"Database validation query failed: {str(e)}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)