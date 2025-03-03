#!/usr/bin/env python
"""
Benchmark Database Maintenance Tool for the IPFS Accelerate Python Framework.

This module provides functions for maintaining the benchmark database, including cleaning
up old JSON files, optimizing the database, and validating data integrity.

Usage:
    python benchmark_db_maintenance.py --clean-json --older-than 30
    python benchmark_db_maintenance.py --optimize
    python benchmark_db_maintenance.py --validate
    python benchmark_db_maintenance.py --backup
"""

import os
import sys
import json
import glob
import shutil
import logging
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

try:
    import duckdb
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkDBMaintenance:
    """
    Maintenance tool for the benchmark database.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark database maintenance tool.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Verify database exists
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
        
        logger.info(f"Initialized BenchmarkDBMaintenance with DB: {db_path}")
    
    def validate_database(self) -> Dict[str, Any]:
        """
        Validate the database structure and data integrity.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "status": "success",
            "errors": [],
            "warnings": [],
            "tables": {},
            "views": {},
            "total_rows": 0
        }
        
        try:
            # Connect to database
            conn = duckdb.connect(self.db_path)
            
            # Check database size
            db_size = os.path.getsize(self.db_path)
            results["db_size"] = db_size
            results["db_size_mb"] = db_size / (1024 * 1024)
            
            # Get list of tables
            tables = conn.execute("SHOW TABLES").fetchall()
            
            if not tables:
                results["status"] = "warning"
                results["warnings"].append("No tables found in database")
                conn.close()
                return results
            
            # Check each table
            total_rows = 0
            
            for table in tables:
                table_name = table[0]
                
                # Get row count
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                total_rows += row_count
                
                # Get schema
                schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
                columns = [col[0] for col in schema]
                
                # Check for primary key
                has_primary_key = any(col.lower() == 'id' or col.lower().endswith('_id') for col in columns)
                
                # Check for timestamp
                has_timestamp = any(col.lower() in ('timestamp', 'created_at', 'updated_at') for col in columns)
                
                # Save table info
                results["tables"][table_name] = {
                    "row_count": row_count,
                    "columns": columns,
                    "has_primary_key": has_primary_key,
                    "has_timestamp": has_timestamp
                }
                
                # Validation checks
                if row_count == 0:
                    results["warnings"].append(f"Table {table_name} is empty")
                    
                if not has_primary_key:
                    results["warnings"].append(f"Table {table_name} doesn't have an obvious primary key")
                    
                if not has_timestamp:
                    results["warnings"].append(f"Table {table_name} doesn't have a timestamp column")
            
            # Get list of views
            views = conn.execute("SHOW VIEWS").fetchall()
            
            for view in views:
                view_name = view[0]
                
                # Get row count from view (may be expensive for complex views)
                try:
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
                    
                    # Save view info
                    results["views"][view_name] = {
                        "row_count": row_count
                    }
                except Exception as e:
                    results["views"][view_name] = {
                        "row_count": "error",
                        "error": str(e)
                    }
                    results["warnings"].append(f"Error counting rows in view {view_name}: {e}")
            
            results["total_rows"] = total_rows
            
            # Overall status
            if results["errors"]:
                results["status"] = "error"
            elif results["warnings"]:
                results["status"] = "warning"
                
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error validating database: {e}")
            results["status"] = "error"
            results["errors"].append(str(e))
            return results
    
    def optimize_database(self) -> bool:
        """
        Optimize the database for better performance.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to database
            conn = duckdb.connect(self.db_path)
            
            # Vacuum the database
            conn.execute("VACUUM")
            logger.info("Database vacuumed")
            
            # Analyze tables for better query planning
            tables = conn.execute("SHOW TABLES").fetchall()
            
            for table in tables:
                table_name = table[0]
                conn.execute(f"ANALYZE {table_name}")
                logger.info(f"Analyzed table: {table_name}")
            
            conn.close()
            
            logger.info("Database optimization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
    
    def backup_database(self, backup_dir: str = "./benchmark_backups") -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_dir: Directory to store backups
            
        Returns:
            Path to the backup file if successful, empty string otherwise
        """
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            db_filename = os.path.basename(self.db_path)
            backup_filename = f"{os.path.splitext(db_filename)[0]}_{timestamp}.duckdb"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy the database file
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return ""
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore the database from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Create a temporary connection to verify backup
            test_conn = duckdb.connect(backup_path, read_only=True)
            test_conn.close()
            
            # Make a backup of the current database before restoring
            current_backup = self.backup_database()
            
            if not current_backup:
                logger.warning("Failed to create backup of current database, proceeding anyway")
                
            # Close any open connections
            # This is a bit of a hack, but DuckDB doesn't provide a direct API to close all connections
            try:
                dummy_conn = duckdb.connect(self.db_path)
                dummy_conn.close()
            except:
                pass
            
            # Copy the backup to the current database
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring database from backup: {e}")
            return False
    
    def clean_json_files(self, directories: List[str], older_than_days: int = 30, 
                         only_if_in_db: bool = True, dry_run: bool = False) -> Dict[str, int]:
        """
        Clean up old JSON files that have been imported to the database.
        
        Args:
            directories: List of directories to scan for JSON files
            older_than_days: Only remove files older than this many days
            only_if_in_db: Only remove files if their data is in the database
            dry_run: Don't actually delete files, just report what would be deleted
            
        Returns:
            Dictionary with counts of files handled
        """
        results = {
            "total_found": 0,
            "eligible_for_deletion": 0,
            "deleted": 0,
            "failed": 0,
            "skipped": 0
        }
        
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        
        for directory in directories:
            if not os.path.isdir(directory):
                logger.warning(f"Directory not found: {directory}")
                continue
                
            # Find all JSON files in the directory
            pattern = os.path.join(directory, "**", "*.json")
            json_files = glob.glob(pattern, recursive=True)
            
            results["total_found"] += len(json_files)
            
            # Process each file
            for file_path in json_files:
                # Check file age
                file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime > cutoff_date:
                    logger.debug(f"Skipping file (too recent): {file_path}")
                    results["skipped"] += 1
                    continue
                
                results["eligible_for_deletion"] += 1
                
                # If only_if_in_db is True, check if the data is in the database
                if only_if_in_db and not self._check_file_in_database(file_path):
                    logger.debug(f"Skipping file (not in database): {file_path}")
                    results["skipped"] += 1
                    continue
                
                # Delete the file
                if not dry_run:
                    try:
                        os.remove(file_path)
                        logger.debug(f"Deleted file: {file_path}")
                        results["deleted"] += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
                        results["failed"] += 1
                else:
                    logger.info(f"Would delete file: {file_path}")
                    results["deleted"] += 1
        
        if dry_run:
            logger.info(f"Dry run completed: would delete {results['deleted']} of {results['total_found']} files")
        else:
            logger.info(f"Cleanup completed: deleted {results['deleted']} of {results['total_found']} files")
        
        return results
    
    def _check_file_in_database(self, file_path: str) -> bool:
        """
        Check if the data from a file is already in the database.
        This is a heuristic check, not perfect.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            True if the data is likely in the database, False otherwise
        """
        try:
            # Get file basename
            file_name = os.path.basename(file_path)
            
            # Load JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Connect to database
            conn = duckdb.connect(self.db_path)
            
            # Try to determine the category from the file
            if "performance" in file_name.lower() or "benchmark" in file_name.lower():
                # Performance data - check if records with this timestamp exist
                if "timestamp" in data:
                    timestamp = data["timestamp"]
                    
                    count = conn.execute(
                        "SELECT COUNT(*) FROM benchmark_performance WHERE source_file = ?",
                        [file_name]
                    ).fetchone()[0]
                    
                    if count > 0:
                        conn.close()
                        return True
            
            elif "hardware" in file_name.lower():
                # Hardware data - check if records with this device name exist
                if "device_name" in data:
                    device_name = data["device_name"]
                    
                    count = conn.execute(
                        "SELECT COUNT(*) FROM benchmark_hardware WHERE source_file = ?",
                        [file_name]
                    ).fetchone()[0]
                    
                    if count > 0:
                        conn.close()
                        return True
            
            elif "compatibility" in file_name.lower():
                # Compatibility data - check if records with this model/hardware exist
                if "model" in data and "hardware_type" in data:
                    model = data["model"]
                    hardware_type = data["hardware_type"]
                    
                    count = conn.execute(
                        "SELECT COUNT(*) FROM benchmark_compatibility WHERE source_file = ?",
                        [file_name]
                    ).fetchone()[0]
                    
                    if count > 0:
                        conn.close()
                        return True
            
            conn.close()
            return False
            
        except Exception as e:
            logger.error(f"Error checking if file {file_path} is in database: {e}")
            return False
    
    def generate_maintenance_report(self, output_file: str = "benchmark_db_maintenance_report.json") -> bool:
        """
        Generate a maintenance report with database statistics and recommendations.
        
        Args:
            output_file: Path to the output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate database
            validation_results = self.validate_database()
            
            # Build report
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "database_path": self.db_path,
                "validation_results": validation_results,
                "recommendations": []
            }
            
            # Add recommendations based on validation results
            if validation_results["status"] == "error":
                report["recommendations"].append({
                    "priority": "high",
                    "message": "Database has errors that should be fixed immediately",
                    "action": "Check error log and fix database issues"
                })
                
            if validation_results["warnings"]:
                report["recommendations"].append({
                    "priority": "medium",
                    "message": f"Database has {len(validation_results['warnings'])} warnings",
                    "action": "Address warnings to improve database integrity"
                })
                
            # Check database size
            db_size_mb = validation_results.get("db_size_mb", 0)
            if db_size_mb > 1000:  # 1 GB
                report["recommendations"].append({
                    "priority": "medium",
                    "message": f"Database size is large ({db_size_mb:.2f} MB)",
                    "action": "Consider cleaning up old data or optimizing the database"
                })
                
            # Check row counts
            for table_name, table_info in validation_results.get("tables", {}).items():
                if table_info["row_count"] > 1000000:  # 1 million rows
                    report["recommendations"].append({
                        "priority": "medium",
                        "message": f"Table {table_name} has {table_info['row_count']} rows",
                        "action": "Consider cleaning up old data from this table"
                    })
            
            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Maintenance report generated: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating maintenance report: {e}")
            return False

def main():
    """Command-line interface for the benchmark database maintenance tool."""
    parser = argparse.ArgumentParser(description="Benchmark Database Maintenance Tool")
    parser.add_argument("--db", default="./benchmark_db.duckdb",
                        help="Path to the DuckDB database")
    parser.add_argument("--validate", action="store_true",
                        help="Validate database structure and integrity")
    parser.add_argument("--optimize", action="store_true",
                        help="Optimize database for better performance")
    parser.add_argument("--backup", action="store_true",
                        help="Create a backup of the database")
    parser.add_argument("--backup-dir", default="./benchmark_backups",
                        help="Directory to store backups")
    parser.add_argument("--restore", 
                        help="Restore database from backup file")
    parser.add_argument("--clean-json", action="store_true",
                        help="Clean up old JSON files")
    parser.add_argument("--dirs", nargs="+", 
                        default=["./archived_test_results", "./performance_results", "./hardware_compatibility_reports"],
                        help="Directories to scan for JSON files")
    parser.add_argument("--older-than", type=int, default=30,
                        help="Only remove files older than this many days")
    parser.add_argument("--only-if-in-db", action="store_true",
                        help="Only remove files if their data is in the database")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually delete files, just report what would be deleted")
    parser.add_argument("--report", action="store_true",
                        help="Generate a maintenance report")
    parser.add_argument("--report-file", default="benchmark_db_maintenance_report.json",
                        help="Path to the maintenance report file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()
    
    # Create maintenance tool
    maintenance = BenchmarkDBMaintenance(db_path=args.db, debug=args.debug)
    
    # Perform requested actions
    if args.validate:
        logger.info("Validating database...")
        results = maintenance.validate_database()
        
        logger.info(f"Validation status: {results['status']}")
        if results["errors"]:
            logger.error(f"Errors ({len(results['errors'])}):")
            for error in results["errors"]:
                logger.error(f"  - {error}")
                
        if results["warnings"]:
            logger.warning(f"Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                logger.warning(f"  - {warning}")
                
        logger.info(f"Tables: {len(results['tables'])}, Views: {len(results['views'])}, Total rows: {results['total_rows']}")
        
    if args.optimize:
        logger.info("Optimizing database...")
        if maintenance.optimize_database():
            logger.info("Database optimization completed successfully")
        else:
            logger.error("Database optimization failed")
            
    if args.backup:
        logger.info("Creating database backup...")
        backup_path = maintenance.backup_database(args.backup_dir)
        
        if backup_path:
            logger.info(f"Database backup created: {backup_path}")
        else:
            logger.error("Database backup failed")
            
    if args.restore:
        logger.info(f"Restoring database from backup: {args.restore}")
        if maintenance.restore_database(args.restore):
            logger.info("Database restoration completed successfully")
        else:
            logger.error("Database restoration failed")
            
    if args.clean_json:
        logger.info(f"Cleaning up old JSON files older than {args.older_than} days...")
        results = maintenance.clean_json_files(
            args.dirs, 
            older_than_days=args.older_than,
            only_if_in_db=args.only_if_in_db,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            logger.info(f"Dry run completed: would delete {results['deleted']} of {results['total_found']} files")
        else:
            logger.info(f"Cleanup completed: deleted {results['deleted']} of {results['total_found']} files")
            
    if args.report:
        logger.info("Generating maintenance report...")
        if maintenance.generate_maintenance_report(args.report_file):
            logger.info(f"Maintenance report generated: {args.report_file}")
        else:
            logger.error("Failed to generate maintenance report")
            
    if not any([args.validate, args.optimize, args.backup, args.restore, args.clean_json, args.report]):
        # No specific action requested, print help
        parser.print_help()

if __name__ == "__main__":
    main()