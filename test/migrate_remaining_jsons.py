#!/usr/bin/env python3
"""
This script migrates any remaining JSON files in the test folder to DuckDB.
It focuses on hardware-related JSON files that should be stored in the database.

Usage:
    python migrate_remaining_jsons.py --db-path ./benchmark_db.duckdb [--delete] [--archive]
"""

import os
import sys
import json
import glob
import argparse
import shutil
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define patterns to migrate
HARDWARE_PATTERNS = [
    "hardware_analysis*.json",
    "hardware_fixes*.json",
    "hardware_detection*.json",
    "hardware_fixed.json",
    "hardware_aware_model_classification.json",
    "parallel_loading_results.json",
    "web_platform_fixes.json",
    "unified_framework_status.json",
    "prediction_test_results.json"
]

def find_json_files(patterns: List[str], directory: str = ".") -> List[str]:
    """Find JSON files matching the given patterns."""
    all_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(directory, pattern))
        all_files.extend(files)
    return all_files

def archive_file(file_path: str, archive_dir: str = "./archived_json_files") -> str:
    """Archive a file by copying it to the archive directory."""
    # Create archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)
    
    # Get the file name
    file_name = os.path.basename(file_path)
    
    # Create archive path with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(archive_dir, f"{file_name}.{timestamp}")
    
    # Copy the file
    shutil.copy2(file_path, archive_path)
    
    return archive_path

def migrate_to_duckdb(file_path: str, db_path: str) -> bool:
    """Migrate a JSON file to DuckDB."""
    try:
        import duckdb
        
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Connect to the database
        conn = duckdb.connect(db_path)
        
        # Determine the appropriate table based on the file name
        file_name = os.path.basename(file_path)
        
        if "hardware_analysis" in file_name:
            # Create model_hardware_analysis table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS model_hardware_analysis (
                timestamp TIMESTAMP,
                model_name VARCHAR,
                model_family VARCHAR,
                analysis_data JSON
            )
            """)
            
            # Extract model name from filename if possible
            model_name = "unknown"
            model_family = "unknown"
            
            # Try to find model name in the file or filename
            if "_hardware_analysis" in file_name:
                model_name = file_name.split("_hardware_analysis")[0]
            elif "model_name" in data:
                model_name = data["model_name"]
            
            # Try to find model family
            if "model_family" in data:
                model_family = data["model_family"]
            
            # Insert the data
            timestamp = duckdb.sql("SELECT now()").fetchone()[0]
            conn.execute(
                "INSERT INTO model_hardware_analysis (timestamp, model_name, model_family, analysis_data) VALUES (?, ?, ?, ?)",
                [timestamp, model_name, model_family, json.dumps(data)]
            )
            
        elif "hardware_fixes" in file_name:
            # Create hardware_fixes table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_fixes (
                timestamp TIMESTAMP,
                files_fixed INTEGER,
                issues_fixed INTEGER,
                data JSON
            )
            """)
            
            # Extract files fixed and issues fixed if possible
            files_fixed = 0
            issues_fixed = 0
            
            if "files_fixed" in data:
                files_fixed = data["files_fixed"]
            if "total_issues_fixed" in data:
                issues_fixed = data["total_issues_fixed"]
                
            # Insert the data
            timestamp = duckdb.sql("SELECT now()").fetchone()[0]
            conn.execute(
                "INSERT INTO hardware_fixes (timestamp, files_fixed, issues_fixed, data) VALUES (?, ?, ?, ?)",
                [timestamp, files_fixed, issues_fixed, json.dumps(data)]
            )
            
        elif "hardware_detection" in file_name:
            # Create hardware_detection table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_detection (
                timestamp TIMESTAMP,
                system_info VARCHAR,
                hardware_data JSON
            )
            """)
            
            # Extract system info if possible
            system_info = "Unknown"
            if "details" in data and "system" in data["details"] and "platform" in data["details"]["system"]:
                system = data["details"]["system"]
                system_info = f"{system.get('platform', '')} {system.get('platform_release', '')} ({system.get('architecture', '')})"
            
            # Insert the data
            timestamp = duckdb.sql("SELECT now()").fetchone()[0]
            conn.execute(
                "INSERT INTO hardware_detection (timestamp, system_info, hardware_data) VALUES (?, ?, ?)",
                [timestamp, system_info, json.dumps(data)]
            )
            
        else:
            # Generic table for other hardware-related files
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_json_files (
                timestamp TIMESTAMP,
                filename VARCHAR,
                data JSON
            )
            """)
            
            # Insert the data
            timestamp = duckdb.sql("SELECT now()").fetchone()[0]
            conn.execute(
                "INSERT INTO hardware_json_files (timestamp, filename, data) VALUES (?, ?, ?)",
                [timestamp, file_name, json.dumps(data)]
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully migrated {file_path} to database")
        return True
    
    except Exception as e:
        logger.error(f"Failed to migrate {file_path} to database: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate JSON files to DuckDB")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to the database")
    parser.add_argument("--delete", action="store_true", help="Delete JSON files after migration")
    parser.add_argument("--archive", action="store_true", help="Archive JSON files before migration")
    parser.add_argument("--archive-dir", type=str, default="./archived_json_files", help="Directory for archived files")
    
    args = parser.parse_args()
    
    # Find JSON files to migrate
    files = find_json_files(HARDWARE_PATTERNS)
    logger.info(f"Found {len(files)} JSON files to migrate")
    
    # Import duckdb
    try:
        import duckdb
    except ImportError:
        logger.error("DuckDB is not installed. Please install it with: pip install duckdb")
        sys.exit(1)
    
    # Check if database exists
    db_path = args.db_path
    if not os.path.exists(db_path):
        logger.info(f"Creating new database at {db_path}")
    
    # Migrate each file
    migrated_count = 0
    archived_count = 0
    deleted_count = 0
    
    for file_path in files:
        logger.info(f"Processing {file_path}...")
        
        # Archive file if requested
        if args.archive:
            archive_path = archive_file(file_path, args.archive_dir)
            logger.info(f"Archived file to {archive_path}")
            archived_count += 1
        
        # Migrate to DuckDB
        success = migrate_to_duckdb(file_path, db_path)
        
        if success:
            migrated_count += 1
            
            # Delete file if requested and migration was successful
            if args.delete and success:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted {file_path}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
    
    # Print summary
    logger.info(f"\nMigration Summary:")
    logger.info(f"- Files found: {len(files)}")
    logger.info(f"- Files migrated: {migrated_count}")
    logger.info(f"- Files archived: {archived_count}")
    logger.info(f"- Files deleted: {deleted_count}")

if __name__ == "__main__":
    main()