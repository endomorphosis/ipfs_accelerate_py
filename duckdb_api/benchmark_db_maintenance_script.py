#!/usr/bin/env python
"""
Benchmark Database Maintenance Tool

This script provides comprehensive maintenance operations for the benchmark database,
including:
- Cleaning up JSON files that have been migrated to the database
- Optimizing database tables and indexes
- Managing data retention policies
- Creating and managing database backups
- Performing integrity checks
- Generating migration statistics
- Purging old backups

Example usage:
  # Optimize database and vacuum to reclaim space
  python benchmark_db_maintenance.py --optimize-db --vacuum --db ./benchmark_db.duckdb
  
  # Create a backup of the database
  python benchmark_db_maintenance.py --backup --backup-dir ./backups
  
  # Clean up JSON files that have been migrated to the database
  python benchmark_db_maintenance.py --clean-json --older-than 30 --action archive
  
  # Run integrity check on the database
  python benchmark_db_maintenance.py --check-integrity
  
  # Generate migration statistics report
  python benchmark_db_maintenance.py --migration-stats --output migration_report.json
"""

import os
import sys
import json
import argparse
import logging
import datetime
import duckdb
import pandas as pd
from pathlib import Path
import glob
import shutil
import hashlib
import time
import re
import subprocess

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_maintenance")

# Constants
DEFAULT_RESULT_DIRS = [
    "performance_results",
    "archived_test_results",
    "hardware_compatibility_reports",
    "collected_results",
    "integration_results",
    "critical_model_results",
    "new_model_results",
    "batch_inference_results"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Maintenance operations for benchmark database")
    
    parser.add_argument("--db", type=str, default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    
    # Maintenance operations
    parser.add_argument("--clean-json", action="store_true",
                        help="Clean up JSON files that have been migrated to the database")
    parser.add_argument("--optimize-db", action="store_true",
                        help="Optimize database tables and indexes")
    parser.add_argument("--vacuum", action="store_true",
                        help="Run VACUUM operation to reclaim space and optimize the database")
    parser.add_argument("--check-integrity", action="store_true",
                        help="Run integrity checks on the database")
    parser.add_argument("--backup", action="store_true",
                        help="Create a backup of the database")
    parser.add_argument("--purge-backups", action="store_true",
                        help="Purge old database backups")
    parser.add_argument("--migration-stats", action="store_true",
                        help="Generate detailed migration statistics")
    
    # Data archiving options
    parser.add_argument("--archive-data", action="store_true",
                        help="Archive old data to separate files for retention")
    parser.add_argument("--archive-dir", type=str, default="./archived_data",
                        help="Directory to store archived data")
    parser.add_argument("--action", type=str, choices=["archive", "remove", "none"], default="none",
                        help="Action to take with processed files (archive, remove, or none)")
    
    # Backup options
    parser.add_argument("--backup-dir", type=str, default="./db_backups",
                        help="Directory to store database backups")
    parser.add_argument("--backup-compress", action="store_true",
                        help="Compress database backups to save space")
    parser.add_argument("--backup-retention", type=int, default=30,
                        help="Number of days to retain database backups")
    
    # Filters and options
    parser.add_argument("--older-than", type=int, default=90,
                        help="Process files/data older than specified days")
    parser.add_argument("--dirs", type=str,
                        help="Comma-separated list of directories to scan for JSON files")
    parser.add_argument("--categories", type=str, default="all",
                        help="Comma-separated list of categories to process (performance,hardware,compatibility,integration,all)")
    parser.add_argument("--output", type=str, 
                        help="Output file for reports (will use stdout if not specified)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--force", action="store_true",
                        help="Force operations without confirmation")
    
    return parser.parse_args()

def connect_to_db(db_path, read_only=False):
    """Connect to the DuckDB database"""
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)
        
    try:
        conn = duckdb.connect(db_path, read_only=read_only)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

def get_processed_files(conn):
    """Get a list of files that have been processed and stored in the database"""
    # First check if migration_tracking table exists
    try:
        table_exists = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'migration_tracking'
        """).fetchone()[0]
        
        if table_exists:
            # Use migration_tracking table from benchmark_db_migration.py
            df = conn.execute("""
            SELECT file_path, file_hash, category, migrated_at, status, records_imported
            FROM migration_tracking
            WHERE status = 'success'
            """).fetchdf()
            
            # Create a dictionary with file path as key
            result = {}
            for _, row in df.iterrows():
                result[row['file_path']] = {
                    'hash': row['file_hash'],
                    'category': row['category'],
                    'migrated_at': row['migrated_at'],
                    'records': row['records_imported']
                }
            
            logger.info(f"Found {len(result)} files in migration_tracking table")
            return result
    except Exception as e:
        logger.debug(f"Error checking migration_tracking table: {e}")
    
    # Fall back to processed_files if migration_tracking doesn't exist
    try:
        # Check if we have a processed_files table
        table_exists = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'processed_files'
        """).fetchone()[0]
        
        if table_exists:
            df = conn.execute("SELECT file_path, processed_at FROM processed_files").fetchdf()
            result = {}
            for _, row in df.iterrows():
                result[row['file_path']] = {
                    'migrated_at': row['processed_at'],
                    'category': 'unknown',
                    'hash': None,
                    'records': 0
                }
            
            logger.info(f"Found {len(result)} files in processed_files table")
            return result
        else:
            # Create the processed_files table if neither exists
            conn.execute("""
            CREATE TABLE processed_files (
                file_id INTEGER PRIMARY KEY,
                file_path VARCHAR NOT NULL UNIQUE,
                file_size INTEGER,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                archived BOOLEAN DEFAULT FALSE,
                archived_at TIMESTAMP
            )
            """)
            
            logger.info("Created processed_files table")
            return {}
    except Exception as e:
        logger.error(f"Error getting processed files: {e}")
        return {}

def find_json_files(directories=None, older_than=90, categories='all'):
    """Find JSON files in specified directories that are older than the given number of days"""
    if directories is None:
        # Use directories relative to the test directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.dirname(script_dir)
        dirs_to_scan = [os.path.join(test_dir, d) for d in DEFAULT_RESULT_DIRS]
    else:
        dirs_to_scan = [d.strip() for d in directories.split(',')]
    
    # Filter by category if specified
    if categories != 'all':
        category_list = [c.strip() for c in categories.split(',')]
    else:
        category_list = None
    
    # Calculate cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than)
    cutoff_timestamp = cutoff_date.timestamp()
    
    json_files = []
    
    for directory in dirs_to_scan:
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
            
        logger.info(f"Scanning for JSON files in: {directory}")
        
        # Find all JSON files in the directory and subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    file_stat = os.stat(file_path)
                    
                    # Check if the file is older than the cutoff date
                    if file_stat.st_mtime < cutoff_timestamp:
                        # Determine the category
                        category = categorize_json_file(file_path)
                        
                        # Check if this category should be processed
                        if category_list is None or category in category_list:
                            json_files.append({
                                'path': file_path,
                                'category': category,
                                'size': file_stat.st_size,
                                'modified': datetime.datetime.fromtimestamp(file_stat.st_mtime)
                            })
    
    logger.info(f"Found {len(json_files)} JSON files older than {older_than} days")
    return json_files

def categorize_json_file(file_path):
    """Determine the category of a JSON file based on naming and content patterns"""
    filename = os.path.basename(file_path)
    dirname = os.path.basename(os.path.dirname(file_path))
    
    # Performance results
    if any(pattern in filename for pattern in ['performance', 'benchmark']):
        return 'performance'
    
    # Hardware compatibility
    if any(pattern in filename for pattern in ['hardware', 'compatibility', 'detection']):
        return 'hardware'
    
    # Model test results
    if any(pattern in filename for pattern in ['model_test', 'critical_model', 'test_status']):
        return 'model'
    
    # Integration test results
    if any(pattern in filename for pattern in ['integration', 'test_results']):
        return 'integration'
    
    # Check file content as a fallback
    try:
        with open(file_path, 'r') as f:
            # Read just the first 1000 chars to detect file type
            content_start = f.read(1000)
            
            if any(key in content_start for key in ['"throughput":', '"latency":', '"memory_peak":']):
                return 'performance'
            
            if any(key in content_start for key in ['"is_compatible":', '"hardware_type":', '"device_name":']):
                return 'hardware'
            
            if any(key in content_start for key in ['"test_name":', '"test_module":', '"status":']):
                return 'integration'
            
            if any(key in content_start for key in ['"model_name":', '"model_family":', '"model_tests":']):
                return 'model'
    except:
        # If we can't read the file, use directory name as a hint
        pass
    
    # If still undetermined, use directory name as hint
    if dirname in ['performance_results', 'benchmark_results', 'batch_inference_results']:
        return 'performance'
    elif dirname in ['hardware_compatibility_reports', 'collected_results']:
        return 'hardware'
    elif dirname in ['integration_results']:
        return 'integration'
    elif dirname in ['critical_model_results', 'new_model_results']:
        return 'model'
    
    # Default to unknown
    return 'unknown'

def is_file_in_database(file_path, processed_files):
    """Check if a file has been processed and stored in the database"""
    return file_path in processed_files

def move_file_to_archive(file_path, archive_dir, preserve_structure=True, dry_run=False):
    """Move a file to the archive directory, preserving its relative path structure if requested"""
    if dry_run:
        logger.info(f"Would move {file_path} to archive")
        return True
    
    try:
        # Create the archive directory if it doesn't exist
        os.makedirs(archive_dir, exist_ok=True)
        
        if preserve_structure:
            # Try to preserve directory structure relative to current directory
            try:
                rel_path = os.path.relpath(file_path, os.getcwd())
                archive_path = os.path.join(archive_dir, rel_path)
            except:
                # Fallback to just the basename
                rel_path = os.path.basename(file_path)
                archive_path = os.path.join(archive_dir, rel_path)
        else:
            # Just use the basename
            rel_path = os.path.basename(file_path)
            archive_path = os.path.join(archive_dir, rel_path)
        
        # If a file with the same name exists, add a timestamp to make it unique
        if os.path.exists(archive_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            base, ext = os.path.splitext(rel_path)
            rel_path = f"{base}_{timestamp}{ext}"
            archive_path = os.path.join(archive_dir, os.path.dirname(rel_path), os.path.basename(rel_path))
        
        # Create any necessary subdirectories in the archive
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        
        # Move the file
        shutil.move(file_path, archive_path)
        logger.debug(f"Moved {file_path} to {archive_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error moving file {file_path} to archive: {e}")
        return False

def delete_file(file_path, dry_run=False):
    """Delete a file from the filesystem"""
    if dry_run:
        logger.info(f"Would delete {file_path}")
        return True
    
    try:
        os.remove(file_path)
        logger.debug(f"Deleted {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False

def update_processed_files(conn, file_info, archived=False, dry_run=False):
    """Update the processed_files table in the database"""
    if dry_run:
        return True
    
    try:
        # Check if we're using migration_tracking or processed_files
        migration_tracking_exists = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'migration_tracking'
        """).fetchone()[0]
        
        if migration_tracking_exists:
            # Use migration_tracking schema
            logger.debug(f"Updating migration_tracking table for {file_info['path']}")
            
            # Check if file exists in migration_tracking
            exists = conn.execute(
                "SELECT COUNT(*) FROM migration_tracking WHERE file_path = ?",
                [file_info['path']]
            ).fetchone()[0]
            
            if exists:
                # Update archived status if needed
                if archived:
                    conn.execute("""
                    UPDATE migration_tracking 
                    SET message = 'File archived: ' || ? 
                    WHERE file_path = ?
                    """, [datetime.datetime.now().isoformat(), file_info['path']])
            else:
                # Calculate hash if needed for new entry
                try:
                    with open(file_info['path'], 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                except Exception as e:
                    logger.error(f"Error calculating hash for {file_info['path']}: {e}")
                    file_hash = "unknown"
                
                # Insert as a new tracked file
                conn.execute("""
                INSERT INTO migration_tracking
                (file_path, file_hash, file_size, category, migrated_at, status, records_imported, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    file_info['path'],
                    file_hash,
                    file_info['size'],
                    file_info.get('category', 'unknown'),
                    datetime.datetime.now(),
                    'archived' if archived else 'pending',
                    0,
                    'File tracked for maintenance' + (' and archived' if archived else '')
                ])
        else:
            # Use processed_files schema
            logger.debug(f"Updating processed_files table for {file_info['path']}")
            
            # Check if the file is already in the table
            exists = conn.execute(
                "SELECT COUNT(*) FROM processed_files WHERE file_path = ?",
                [file_info['path']]
            ).fetchone()[0]
            
            if exists:
                # Update the existing record
                if archived:
                    conn.execute(
                        "UPDATE processed_files SET archived = TRUE, archived_at = ? WHERE file_path = ?",
                        [datetime.datetime.now(), file_info['path']]
                    )
            else:
                # Insert a new record
                conn.execute(
                    "INSERT INTO processed_files (file_path, file_size, processed_at, archived, archived_at) VALUES (?, ?, ?, ?, ?)",
                    [
                        file_info['path'],
                        file_info['size'],
                        file_info['modified'],
                        archived,
                        datetime.datetime.now() if archived else None
                    ]
                )
        
        return True
    except Exception as e:
        logger.error(f"Error updating tracking table: {e}")
        return False

def clean_json_files(conn, args):
    """Clean up JSON files that have been migrated to the database"""
    processed_files = get_processed_files(conn)
    json_files = find_json_files(args.dirs, args.older_than, args.categories)
    
    files_archived = 0
    files_deleted = 0
    files_skipped = 0
    
    for file_info in json_files:
        file_path = file_info['path']
        
        # Check if the file has been processed
        if is_file_in_database(file_path, processed_files):
            logger.info(f"File has been processed: {file_path}")
            
            if args.action == "archive":
                # Move to archive
                if move_file_to_archive(file_path, args.archive_dir, True, args.dry_run):
                    files_archived += 1
                    if not args.dry_run:
                        update_processed_files(conn, file_info, archived=True)
            elif args.action == "remove":
                # Delete the file
                if delete_file(file_path, args.dry_run):
                    files_deleted += 1
                    if not args.dry_run:
                        update_processed_files(conn, file_info, archived=True)
        else:
            logger.debug(f"File not yet processed in database: {file_path}")
            files_skipped += 1
    
    logger.info(f"Clean JSON operation summary:")
    logger.info(f"  - Files archived: {files_archived}")
    logger.info(f"  - Files deleted: {files_deleted}")
    logger.info(f"  - Files skipped (not in database): {files_skipped}")
    
    return {
        "archived": files_archived,
        "deleted": files_deleted,
        "skipped": files_skipped,
        "total_processed": files_archived + files_deleted,
        "total_files": files_archived + files_deleted + files_skipped
    }

def optimize_database(conn, args):
    """Optimize database tables and indexes"""
    if args.dry_run:
        logger.info("Would optimize database tables and indexes")
        return {"optimized_tables": 0, "optimized_indexes": 0}
    
    try:
        # Get list of tables
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        # Track statistics
        optimized_tables = 0
        optimized_indexes = 0
        table_stats = {}
        
        for table in table_names:
            logger.info(f"Optimizing table: {table}")
            
            # Get row count before optimization
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            
            # Record stats before optimization
            table_stats[table] = {
                "rows": row_count,
                "indexes": []
            }
            
            # Analyze the table to update statistics
            start_time = time.time()
            conn.execute(f"ANALYZE {table}")
            analyze_time = time.time() - start_time
            
            # Check if the table has indexes
            indexes = conn.execute(f"PRAGMA indexes({table})").fetchall()
            
            if indexes:
                logger.info(f"  - Found {len(indexes)} indexes")
                
                # Get index details
                for idx in indexes:
                    index_name = idx[0]
                    
                    # Get columns in the index
                    try:
                        index_cols = conn.execute(f"PRAGMA index_info({index_name})").fetchall()
                        index_columns = [col[2] for col in index_cols]  # column names
                        
                        table_stats[table]["indexes"].append({
                            "name": index_name,
                            "columns": index_columns
                        })
                    except:
                        table_stats[table]["indexes"].append({
                            "name": index_name,
                            "columns": []
                        })
                
                # Reindex to optimize the indexes
                start_time = time.time()
                conn.execute(f"REINDEX {table}")
                reindex_time = time.time() - start_time
                
                optimized_indexes += len(indexes)
                table_stats[table]["reindex_time"] = reindex_time
            
            optimized_tables += 1
            table_stats[table]["analyze_time"] = analyze_time
        
        logger.info(f"Database optimization completed: {optimized_tables} tables, {optimized_indexes} indexes")
        
        return {
            "optimized_tables": optimized_tables,
            "optimized_indexes": optimized_indexes,
            "table_stats": table_stats
        }
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        return {"error": str(e)}

def vacuum_database(conn, args):
    """Run VACUUM to reclaim space and optimize the database"""
    if args.dry_run:
        logger.info("Would run VACUUM on the database")
        return {"status": "skipped (dry run)"}
    
    try:
        # Get database size before vacuum
        db_size_before = os.path.getsize(args.db)
        logger.info(f"Database size before VACUUM: {db_size_before / (1024*1024):.2f} MB")
        
        # Run VACUUM
        logger.info("Running VACUUM (this may take a while)...")
        start_time = time.time()
        conn.execute("VACUUM")
        vacuum_time = time.time() - start_time
        
        # Get database size after vacuum
        db_size_after = os.path.getsize(args.db)
        logger.info(f"Database size after VACUUM: {db_size_after / (1024*1024):.2f} MB")
        
        # Report space saved
        space_saved = db_size_before - db_size_after
        if space_saved > 0:
            logger.info(f"Space saved: {space_saved / (1024*1024):.2f} MB ({space_saved / db_size_before * 100:.2f}%)")
        else:
            logger.info("No space saved by VACUUM operation")
        
        return {
            "status": "success",
            "size_before_mb": round(db_size_before / (1024*1024), 2),
            "size_after_mb": round(db_size_after / (1024*1024), 2),
            "space_saved_mb": round(space_saved / (1024*1024), 2) if space_saved > 0 else 0,
            "space_saved_percent": round(space_saved / db_size_before * 100, 2) if space_saved > 0 else 0,
            "vacuum_time_seconds": round(vacuum_time, 2)
        }
    except Exception as e:
        logger.error(f"Error during VACUUM operation: {e}")
        return {"status": "error", "error": str(e)}

def archive_old_data(conn, args):
    """Archive old data from the database for retention"""
    if args.dry_run:
        logger.info("Would archive old data from the database")
        return {"status": "skipped (dry run)"}
    
    try:
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=args.older_than)
        
        # Create archive directory
        os.makedirs(args.archive_dir, exist_ok=True)
        
        # Archive timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine tables to archive
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        # Tables with timestamps that we should consider for archiving
        tables_to_check = {}
        tables_archived = []
        total_rows_archived = 0
        
        # Identify tables with timestamp columns
        for table in table_names:
            try:
                # Get columns for this table
                columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
                col_names = [col[1] for col in columns]
                
                # Check for timestamp columns using common naming patterns
                timestamp_cols = [
                    col for col in col_names 
                    if col.lower() in ('created_at', 'timestamp', 'run_date', 'test_date', 'started_at', 'executed_at', 'migrated_at')
                ]
                
                if timestamp_cols:
                    tables_to_check[table] = timestamp_cols[0]  # Use the first timestamp column
            except Exception as e:
                logger.debug(f"Error checking columns for table {table}: {e}")
        
        # Archive old data from each table
        for table, timestamp_col in tables_to_check.items():
            logger.info(f"Checking for old data in {table} (using {timestamp_col} column)")
            
            # Query to check for old data
            try:
                count_query = f"""
                SELECT COUNT(*) FROM {table}
                WHERE {timestamp_col} < ?
                """
                
                old_data_count = conn.execute(count_query, [cutoff_date]).fetchone()[0]
                
                if old_data_count > 0:
                    logger.info(f"  - Found {old_data_count} rows older than {args.older_than} days")
                    
                    # Query to get the old data
                    data_query = f"""
                    SELECT * FROM {table}
                    WHERE {timestamp_col} < ?
                    """
                    
                    old_data = conn.execute(data_query, [cutoff_date]).fetchdf()
                    
                    # Save to Parquet file
                    archive_file = os.path.join(args.archive_dir, f"{table}_archive_{timestamp}.parquet")
                    old_data.to_parquet(archive_file)
                    
                    logger.info(f"  - Archived {len(old_data)} rows to {archive_file}")
                    total_rows_archived += len(old_data)
                    tables_archived.append({
                        "table": table,
                        "rows": len(old_data),
                        "archive_file": archive_file,
                        "date_column": timestamp_col,
                        "cutoff_date": cutoff_date.isoformat()
                    })
                    
                    # Optional: Delete the archived data
                    if args.action == "remove":
                        delete_query = f"""
                        DELETE FROM {table}
                        WHERE {timestamp_col} < ?
                        """
                        
                        conn.execute(delete_query, [cutoff_date])
                        logger.info(f"  - Deleted {len(old_data)} rows from {table}")
                else:
                    logger.info(f"  - No data older than {args.older_than} days found")
            except Exception as e:
                logger.warning(f"  - Error processing table {table}: {e}")
        
        if total_rows_archived > 0:
            logger.info(f"Data archiving completed: {total_rows_archived} total rows archived from {len(tables_archived)} tables")
        else:
            logger.info("No data found to archive")
        
        return {
            "status": "success",
            "tables_archived": len(tables_archived),
            "total_rows_archived": total_rows_archived,
            "table_details": tables_archived,
            "archive_dir": args.archive_dir,
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error archiving old data: {e}")
        return {"status": "error", "error": str(e)}

def backup_database(conn, args):
    """Create a backup of the database"""
    if args.dry_run:
        logger.info(f"Would backup database {args.db} to {args.backup_dir}")
        return {"status": "skipped (dry run)"}
    
    try:
        # Create backup directory
        os.makedirs(args.backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        db_name = os.path.basename(args.db)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{os.path.splitext(db_name)[0]}_backup_{timestamp}.duckdb"
        backup_path = os.path.join(args.backup_dir, backup_name)
        
        # Get database size before backup
        db_size = os.path.getsize(args.db)
        
        # Close the connection to ensure all changes are flushed
        conn.close()
        logger.info("Closed database connection for backup")
        
        # Make the backup
        logger.info(f"Creating backup at {backup_path}...")
        start_time = time.time()
        
        if args.backup_compress:
            # Use compression for the backup
            backup_path_compressed = f"{backup_path}.gz"
            
            # First copy the file
            shutil.copy2(args.db, backup_path)
            
            # Then compress it
            try:
                import gzip
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(backup_path_compressed, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove the uncompressed copy
                os.remove(backup_path)
                backup_path = backup_path_compressed
            except ImportError:
                logger.warning("gzip module not available, creating uncompressed backup")
        else:
            # Simple file copy for uncompressed backup
            shutil.copy2(args.db, backup_path)
        
        backup_time = time.time() - start_time
        
        # Get backup size
        backup_size = os.path.getsize(backup_path)
        compression_ratio = (1 - (backup_size / db_size)) * 100 if args.backup_compress else 0
        
        logger.info(f"Backup completed in {backup_time:.2f} seconds")
        logger.info(f"Original size: {db_size / (1024*1024):.2f} MB, Backup size: {backup_size / (1024*1024):.2f} MB")
        
        if args.backup_compress:
            logger.info(f"Compression ratio: {compression_ratio:.2f}%")
        
        return {
            "status": "success",
            "backup_file": backup_path,
            "backup_time_seconds": round(backup_time, 2),
            "original_size_mb": round(db_size / (1024*1024), 2),
            "backup_size_mb": round(backup_size / (1024*1024), 2),
            "compression_ratio": round(compression_ratio, 2) if args.backup_compress else 0,
            "compressed": args.backup_compress,
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        # Reconnect to the database
        conn = connect_to_db(args.db)

def purge_old_backups(args):
    """Purge old database backups"""
    if args.dry_run:
        logger.info(f"Would purge database backups older than {args.backup_retention} days from {args.backup_dir}")
        return {"status": "skipped (dry run)"}
    
    try:
        if not os.path.exists(args.backup_dir):
            logger.warning(f"Backup directory {args.backup_dir} does not exist")
            return {"status": "skipped", "reason": "backup directory does not exist"}
        
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=args.backup_retention)
        cutoff_timestamp = cutoff_date.timestamp()
        
        # Find backup files
        backup_files = []
        backup_pattern = re.compile(r'.*_backup_(\d{8}_\d{6})\.duckdb(\.gz)?$')
        
        for file in os.listdir(args.backup_dir):
            file_path = os.path.join(args.backup_dir, file)
            if os.path.isfile(file_path):
                match = backup_pattern.match(file)
                if match:
                    file_stat = os.stat(file_path)
                    if file_stat.st_mtime < cutoff_timestamp:
                        backup_files.append({
                            'path': file_path,
                            'size': file_stat.st_size,
                            'modified': datetime.datetime.fromtimestamp(file_stat.st_mtime),
                            'timestamp_str': match.group(1)
                        })
        
        # Delete old backups
        deleted_count = 0
        deleted_size = 0
        deleted_files = []
        
        for backup in backup_files:
            logger.info(f"Deleting old backup: {backup['path']} ({backup['size'] / (1024*1024):.2f} MB, from {backup['modified'].strftime('%Y-%m-%d %H:%M:%S')})")
            os.remove(backup['path'])
            deleted_count += 1
            deleted_size += backup['size']
            deleted_files.append(backup['path'])
        
        if deleted_count > 0:
            logger.info(f"Purged {deleted_count} backups, freed {deleted_size / (1024*1024):.2f} MB of space")
        else:
            logger.info(f"No backups older than {args.backup_retention} days found")
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "deleted_size_mb": round(deleted_size / (1024*1024), 2),
            "deleted_files": deleted_files,
            "cutoff_date": cutoff_date.isoformat()
        }
    except Exception as e:
        logger.error(f"Error purging old backups: {e}")
        return {"status": "error", "error": str(e)}

def check_database_integrity(conn, args):
    """Check database integrity"""
    if args.dry_run:
        logger.info("Would check database integrity")
        return {"status": "skipped (dry run)"}
    
    try:
        logger.info("Running database integrity checks...")
        issues_found = []
        checks_performed = []
        
        # 1. Check if all tables can be queried
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        logger.info(f"Checking {len(table_names)} tables for basic integrity")
        
        checks_performed.append("table_existence")
        
        for table in table_names:
            try:
                # Check if we can query the table
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                logger.debug(f"  - Table {table}: {count} rows")
            except Exception as e:
                issues_found.append({
                    "type": "table_query_failed",
                    "table": table,
                    "error": str(e)
                })
        
        # 2. Check foreign key constraints
        logger.info("Checking foreign key constraints")
        checks_performed.append("foreign_keys")
        
        # Get tables with foreign keys
        fk_relations = []
        
        for table in table_names:
            try:
                # Look for foreign key columns
                pragmas = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
                
                for fk in pragmas:
                    fk_relations.append({
                        "table": table,
                        "column": fk[3],
                        "references_table": fk[2],
                        "references_column": fk[4]
                    })
            except Exception as e:
                logger.debug(f"Error checking foreign keys for {table}: {e}")
        
        logger.info(f"Found {len(fk_relations)} foreign key relationships")
        
        # Check each foreign key relationship for orphaned records
        for relation in fk_relations:
            try:
                query = f"""
                SELECT COUNT(*) FROM {relation['table']} t
                LEFT JOIN {relation['references_table']} r ON t.{relation['column']} = r.{relation['references_column']}
                WHERE t.{relation['column']} IS NOT NULL AND r.{relation['references_column']} IS NULL
                """
                
                orphaned_count = conn.execute(query).fetchone()[0]
                
                if orphaned_count > 0:
                    issues_found.append({
                        "type": "orphaned_foreign_key",
                        "relation": relation,
                        "count": orphaned_count,
                        "query": query
                    })
                    
                    logger.warning(f"Found {orphaned_count} orphaned records in {relation['table']} referencing {relation['references_table']}")
            except Exception as e:
                logger.debug(f"Error checking foreign key relationship {relation}: {e}")
        
        # 3. Check for duplicate primary keys
        logger.info("Checking for duplicate primary keys")
        checks_performed.append("primary_keys")
        
        for table in table_names:
            try:
                # Get primary key columns
                pragmas = conn.execute(f"PRAGMA table_info({table})").fetchall()
                pk_columns = [col[1] for col in pragmas if col[5] == 1]  # column 5 is the PK flag
                
                if pk_columns:
                    logger.debug(f"  - Table {table} has primary key on columns: {', '.join(pk_columns)}")
                    
                    # Check for duplicate values in primary key
                    pk_cols_str = ", ".join(pk_columns)
                    query = f"""
                    SELECT {pk_cols_str}, COUNT(*) as count 
                    FROM {table} 
                    GROUP BY {pk_cols_str} 
                    HAVING COUNT(*) > 1
                    """
                    
                    duplicates = conn.execute(query).fetchall()
                    
                    if duplicates:
                        for dup in duplicates:
                            issues_found.append({
                                "type": "duplicate_primary_key",
                                "table": table,
                                "primary_key_columns": pk_columns,
                                "duplicate_count": dup[-1],
                                "duplicate_values": dup[:-1]
                            })
                        
                        logger.warning(f"Found {len(duplicates)} duplicate primary key values in {table}")
            except Exception as e:
                logger.debug(f"Error checking primary keys for {table}: {e}")
        
        # 4. Check for NULL values in non-NULL columns
        logger.info("Checking for NULL values in non-NULL columns")
        checks_performed.append("null_constraints")
        
        for table in table_names:
            try:
                # Get columns that should not be NULL
                pragmas = conn.execute(f"PRAGMA table_info({table})").fetchall()
                not_null_columns = [col[1] for col in pragmas if col[3] == 1]  # column 3 is NOT NULL flag
                
                for column in not_null_columns:
                    query = f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE {column} IS NULL
                    """
                    
                    null_count = conn.execute(query).fetchone()[0]
                    
                    if null_count > 0:
                        issues_found.append({
                            "type": "null_in_not_null_column",
                            "table": table,
                            "column": column,
                            "count": null_count
                        })
                        
                        logger.warning(f"Found {null_count} NULL values in non-NULL column {table}.{column}")
            except Exception as e:
                logger.debug(f"Error checking NULL constraints for {table}: {e}")
        
        # 5. Check for missing or inconsistent data
        logger.info("Checking for data consistency")
        checks_performed.append("data_consistency")
        
        # Define expected relationships
        expected_relationships = [
            {
                "description": "All performance results should have valid models",
                "query": """
                SELECT COUNT(*) FROM performance_results pr
                LEFT JOIN models m ON pr.model_id = m.model_id
                WHERE m.model_id IS NULL
                """
            },
            {
                "description": "All performance results should have valid hardware platforms",
                "query": """
                SELECT COUNT(*) FROM performance_results pr
                LEFT JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE hp.hardware_id IS NULL
                """
            },
            {
                "description": "All performance results should have valid test runs",
                "query": """
                SELECT COUNT(*) FROM performance_results pr
                LEFT JOIN test_runs tr ON pr.run_id = tr.run_id
                WHERE tr.run_id IS NULL
                """
            }
        ]
        
        for relationship in expected_relationships:
            try:
                inconsistent_count = conn.execute(relationship["query"]).fetchone()[0]
                
                if inconsistent_count > 0:
                    issues_found.append({
                        "type": "data_inconsistency",
                        "description": relationship["description"],
                        "count": inconsistent_count,
                        "query": relationship["query"]
                    })
                    
                    logger.warning(f"Data inconsistency: {relationship['description']} - found {inconsistent_count} issues")
            except Exception as e:
                logger.debug(f"Error checking data consistency: {e}")
        
        # Summary
        if issues_found:
            logger.warning(f"Found {len(issues_found)} issues during integrity check")
        else:
            logger.info("No integrity issues found")
        
        return {
            "status": "success" if not issues_found else "issues_found",
            "checks_performed": checks_performed,
            "issues_count": len(issues_found),
            "issues": issues_found
        }
    except Exception as e:
        logger.error(f"Error checking database integrity: {e}")
        return {"status": "error", "error": str(e)}

def generate_migration_statistics(conn, args):
    """Generate detailed migration statistics"""
    try:
        logger.info("Generating migration statistics...")
        
        # Check if migration tracking table exists
        migration_tracking_exists = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'migration_tracking'
        """).fetchone()[0]
        
        if not migration_tracking_exists:
            logger.warning("Migration tracking table not found, cannot generate statistics")
            return {"status": "error", "error": "Migration tracking table not found"}
        
        # Get basic statistics
        basic_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_files,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_files,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_files,
            SUM(records_imported) as total_records_imported,
            SUM(file_size) / (1024*1024) as total_size_mb
        FROM migration_tracking
        """).fetchdf().iloc[0].to_dict()
        
        # Get statistics by category
        category_stats = conn.execute("""
        SELECT 
            category,
            COUNT(*) as total_files,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_files,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_files,
            SUM(records_imported) as total_records_imported,
            SUM(file_size) / (1024*1024) as total_size_mb
        FROM migration_tracking
        GROUP BY category
        ORDER BY total_files DESC
        """).fetchdf().to_dict('records')
        
        # Get statistics over time (by day)
        time_stats = conn.execute("""
        SELECT 
            CAST(migrated_at AS DATE) as migration_date,
            COUNT(*) as files_processed,
            SUM(records_imported) as records_imported
        FROM migration_tracking
        GROUP BY CAST(migrated_at AS DATE)
        ORDER BY CAST(migrated_at AS DATE)
        """).fetchdf().to_dict('records')
        
        # Get the latest migrations
        latest_migrations = conn.execute("""
        SELECT 
            file_path, 
            category, 
            status, 
            records_imported,
            migrated_at,
            message
        FROM migration_tracking
        ORDER BY migrated_at DESC
        LIMIT 10
        """).fetchdf().to_dict('records')
        
        # Database statistics
        table_stats = conn.execute("""
        SELECT 
            table_name,
            COUNT(*) as approx_row_count
        FROM (
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name NOT IN ('migration_tracking', 'processed_files')
        ) t
        LEFT JOIN (
            SELECT table_name, COUNT(*) as table_count
            FROM information_schema.tables
            GROUP BY table_name
        ) c ON t.table_name = c.table_name
        GROUP BY table_name
        """).fetchdf().to_dict('records')
        
        # Combine all statistics
        stats = {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "database_file": args.db,
            "basic_stats": basic_stats,
            "category_stats": category_stats,
            "time_stats": time_stats,
            "latest_migrations": latest_migrations,
            "table_stats": table_stats
        }
        
        # Generate a report if output file is specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Migration statistics written to {args.output}")
        
        logger.info("Migration statistics summary:")
        logger.info(f"  - Total files processed: {basic_stats['total_files']}")
        logger.info(f"  - Successful files: {basic_stats['successful_files']}")
        logger.info(f"  - Failed files: {basic_stats['failed_files']}")
        logger.info(f"  - Total records imported: {basic_stats['total_records_imported']}")
        logger.info(f"  - Total file size: {basic_stats['total_size_mb']:.2f} MB")
        
        return stats
    except Exception as e:
        logger.error(f"Error generating migration statistics: {e}")
        return {"status": "error", "error": str(e)}

def main():
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Track results for all operations
    results = {}
    
    # Handle backup operation first since it will disconnect
    if args.backup:
        # Connect in read-only mode for backup
        conn = connect_to_db(args.db, read_only=True)
        logger.info(f"Creating backup of database {args.db}")
        results['backup'] = backup_database(conn, args)
        conn.close()
    
    # Purge old backups (doesn't need database connection)
    if args.purge_backups:
        logger.info(f"Purging old database backups from {args.backup_dir}")
        results['purge_backups'] = purge_old_backups(args)
    
    # Connect to the database for other operations
    if any([
        args.clean_json, 
        args.optimize_db, 
        args.archive_data, 
        args.vacuum, 
        args.check_integrity,
        args.migration_stats
    ]):
        conn = connect_to_db(args.db)
        
        # Perform requested maintenance operations
        operations_performed = 0
        
        if args.clean_json:
            logger.info("Cleaning up JSON files that have been migrated to the database")
            results['clean_json'] = clean_json_files(conn, args)
            operations_performed += 1
        
        if args.optimize_db:
            logger.info("Optimizing database tables and indexes")
            results['optimize_db'] = optimize_database(conn, args)
            operations_performed += 1
        
        if args.archive_data:
            logger.info(f"Archiving data older than {args.older_than} days")
            results['archive_data'] = archive_old_data(conn, args)
            operations_performed += 1
        
        if args.check_integrity:
            logger.info("Checking database integrity")
            results['check_integrity'] = check_database_integrity(conn, args)
            operations_performed += 1
        
        if args.migration_stats:
            logger.info("Generating migration statistics")
            results['migration_stats'] = generate_migration_statistics(conn, args)
            operations_performed += 1
        
        if args.vacuum:
            # Vacuum should be run last after other operations
            logger.info("Running VACUUM operation on the database")
            results['vacuum'] = vacuum_database(conn, args)
            operations_performed += 1
        
        if operations_performed == 0:
            logger.warning("No maintenance operations specified. Use --help to see available options.")
        
        # Commit changes and close the database connection
        conn.commit()
        conn.close()
    
    # Summary of all operations
    logger.info("\nMaintenance operations summary:")
    for operation, result in results.items():
        if isinstance(result, dict) and 'status' in result:
            logger.info(f"  - {operation}: {result['status']}")
        else:
            logger.info(f"  - {operation}: completed")
    
    # Output full results if requested
    if args.output and 'migration_stats' not in results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results written to {args.output}")
    
    logger.info("All maintenance operations completed")

if __name__ == "__main__":
    main()