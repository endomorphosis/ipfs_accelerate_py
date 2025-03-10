#!/usr/bin/env python
"""
Migrate All JSON Files to DuckDB Script

This script finds all JSON benchmark files in the codebase and migrates them to
the DuckDB database. It uses the benchmark_db_converter.py tool to perform the
migration. After successful migration, JSON files can be optionally archived.

Usage:
    python migrate_all_json_files.py --db-path ./benchmark_db.duckdb
    python migrate_all_json_files.py --db-path ./benchmark_db.duckdb --archive
    python migrate_all_json_files.py --db-path ./benchmark_db.duckdb --delete
"""

import os
import sys
import subprocess
import argparse
import logging
import datetime
import tarfile
import shutil
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_json_files(base_dir='.', exclude_dirs=None):
    """
    Find all JSON files in the specified directory and its subdirectories.
    Excludes specified directories.
    
    Args:
        base_dir: Base directory to search
        exclude_dirs: List of directories to exclude from search
        
    Returns:
        List of JSON file paths
    """
    if exclude_dirs is None:
        exclude_dirs = ['node_modules', '.git', 'venv', 'env']
    
    json_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return json_files

def group_files_by_directory(json_files):
    """
    Group files by their parent directory for more efficient migration.
    
    Args:
        json_files: List of JSON file paths
        
    Returns:
        Dictionary of directory paths to lists of file paths
    """
    grouped = {}
    for file_path in json_files:
        parent_dir = os.path.dirname(file_path)
        if parent_dir not in grouped:
            grouped[parent_dir] = []
        grouped[parent_dir].append(file_path)
    
    return grouped

def migrate_json_files(directories, db_path):
    """
    Migrate JSON files to DuckDB using benchmark_db_converter.py.
    
    Args:
        directories: Dictionary of directories to list of files
        db_path: Path to the DuckDB database
        
    Returns:
        Dictionary mapping directories to success status
    """
    results = {}
    timestamp = int(time.time())
    temp_db_path = f"{db_path.rsplit('.', 1)[0]}_{timestamp}.duckdb"
    
    # Check if the database is locked
    db_is_locked = False
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=False)
        con.close()
    except Exception as e:
        if "lock" in str(e).lower():
            db_is_locked = True
            logger.warning(f"Database {db_path} is locked. Using temporary database {temp_db_path}")
    
    # Use the temporary database if the original is locked
    actual_db_path = temp_db_path if db_is_locked else db_path
    
    successful_directories = []
    
    for directory, files in directories.items():
        logger.info(f"Migrating {len(files)} files from {directory}")
        
        # Skip if no JSON files in directory
        if not files:
            continue
        
        # Call benchmark_db_converter.py to migrate files
        cmd = [
            "python", "benchmark_db_converter.py",
            "--input-dir", directory,
            "--output-db", actual_db_path,
            "--deduplicate"
        ]
        
        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            results[directory] = True
            successful_directories.append(directory)
            logger.info(f"Successfully migrated files from {directory}")
        except subprocess.CalledProcessError as e:
            results[directory] = False
            error_output = e.stderr if e.stderr else e.stdout
            if "lock" in error_output.lower():
                logger.error(f"Database lock error migrating files from {directory}. Will try again with a new database.")
                # Retry with a new database name
                new_timestamp = int(time.time())
                retry_db_path = f"{db_path.rsplit('.', 1)[0]}_{new_timestamp}.duckdb"
                cmd[4] = retry_db_path
                try:
                    subprocess.run(cmd, check=True)
                    results[directory] = True
                    successful_directories.append(directory)
                    logger.info(f"Successfully migrated files from {directory} to {retry_db_path}")
                except subprocess.CalledProcessError as retry_error:
                    results[directory] = False
                    logger.error(f"Error migrating files from {directory} on retry: {retry_error}")
            else:
                logger.error(f"Error migrating files from {directory}: {e}")
    
    if db_is_locked and successful_directories:
        logger.warning(f"Data was saved to {actual_db_path} because the original database was locked.")
        logger.warning(f"You will need to merge the databases later using benchmark_db_maintenance.py.")
    
    return results

def archive_json_files(json_files, archive_dir="./archived_json_files"):
    """
    Archive JSON files into a compressed tarfile.
    
    Args:
        json_files: List of JSON file paths
        archive_dir: Directory for storing the archive
        
    Returns:
        Path to the created archive file
    """
    # Create archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)
    
    # Create archive filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_file = os.path.join(archive_dir, f"json_files_{timestamp}.tar.gz")
    
    # Create tarfile
    with tarfile.open(archive_file, "w:gz") as tar:
        for file_path in json_files:
            # Add file to archive with relative path
            tar.add(file_path, arcname=os.path.relpath(file_path))
    
    logger.info(f"Archived {len(json_files)} JSON files to {archive_file}")
    return archive_file

def delete_json_files(json_files):
    """
    Delete JSON files after archiving.
    
    Args:
        json_files: List of JSON file paths
        
    Returns:
        Number of successfully deleted files
    """
    deleted_count = 0
    for file_path in json_files:
        try:
            os.remove(file_path)
            deleted_count += 1
        except OSError as e:
            logger.error(f"Error deleting {file_path}: {e}")
    
    logger.info(f"Deleted {deleted_count} out of {len(json_files)} JSON files")
    return deleted_count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Migrate all JSON files to DuckDB database")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                      help="Path to the DuckDB database file")
    parser.add_argument("--archive", action="store_true",
                      help="Archive JSON files after migration")
    parser.add_argument("--archive-dir", default="./archived_json_files",
                      help="Directory for storing the archive")
    parser.add_argument("--delete", action="store_true",
                      help="Delete JSON files after migration (implies --archive)")
    parser.add_argument("--exclude-dirs", nargs="+", default=["node_modules", ".git", "venv", "env"],
                      help="Directories to exclude from migration")
    parser.add_argument("--base-dir", default=".",
                      help="Base directory to search for JSON files")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # If delete is specified, always archive first
    if args.delete:
        args.archive = True
    
    # Find all JSON files
    logger.info(f"Searching for JSON files in {args.base_dir}...")
    json_files = find_json_files(args.base_dir, args.exclude_dirs)
    logger.info(f"Found {len(json_files)} JSON files")
    
    if not json_files:
        logger.info("No JSON files found. Nothing to migrate.")
        return
    
    # Group files by directory
    grouped_files = group_files_by_directory(json_files)
    logger.info(f"Files grouped into {len(grouped_files)} directories")
    
    # Migrate files to DuckDB
    logger.info(f"Migrating JSON files to DuckDB database at {args.db_path}...")
    migration_results = migrate_json_files(grouped_files, args.db_path)
    
    # Count successful migrations
    successful_dirs = sum(1 for success in migration_results.values() if success)
    logger.info(f"Successfully migrated files from {successful_dirs} out of {len(grouped_files)} directories")
    
    # Archive JSON files if requested
    if args.archive:
        logger.info(f"Archiving JSON files to {args.archive_dir}...")
        archive_file = archive_json_files(json_files, args.archive_dir)
        logger.info(f"JSON files archived to {archive_file}")
    
    # Delete JSON files if requested
    if args.delete:
        logger.info("Deleting JSON files...")
        deleted_count = delete_json_files(json_files)
        logger.info(f"Deleted {deleted_count} JSON files")
    
    logger.info("Migration process completed")

if __name__ == "__main__":
    main()