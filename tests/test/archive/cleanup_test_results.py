#!/usr/bin/env python
"""
Test Results Cleanup and Migration Tool

This script cleans up test result JSON files by migrating them to the DuckDB/Parquet database
and optionally archiving or removing the original files. It's designed to be run periodically
to keep the repository clean while preserving all test data.
"""

import os
import sys
import json
import argparse
import logging
import datetime
import shutil
from pathlib import Path
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cleanup_tool")

def parse_args():
    parser = argparse.ArgumentParser(description="Clean up test result JSON files")
    
    parser.add_argument("--db", type=str, default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    parser.add_argument("--action", type=str, choices=['archive', 'remove', 'simulate'], default='archive',
                        help="Action to take with processed files (archive, remove, or simulate)")
    parser.add_argument("--archive-dir", type=str, default="./archived_results",
                        help="Directory to archive processed files")
    parser.add_argument("--older-than", type=int, default=30,
                        help="Only process files older than N days")
    parser.add_argument("--categories", type=str, default="all",
                        help="Comma-separated list of categories to process (performance,hardware,compatibility,integration,all)")
    parser.add_argument("--exclude-dirs", type=str,
                        help="Comma-separated list of directories to exclude")
    parser.add_argument("--include-dirs", type=str,
                        help="Comma-separated list of directories to include (overrides automatic discovery)")
    parser.add_argument("--max-files", type=int, default=1000,
                        help="Maximum number of files to process in one run")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel processes for conversion")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Skip files that cause errors instead of stopping")
    parser.add_argument("--confirm", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    return parser.parse_args()

def find_json_files(args):
    """Find JSON test result files to process"""
    # Determine directories to scan
    if args.include_dirs:
        dirs_to_scan = [d.strip() for d in args.include_dirs.split(',')]
    else:
        # Default directories to scan
        default_dirs = [
            "performance_results",
            "archived_test_results",
            "hardware_compatibility_reports",
            "collected_results",
            "integration_results",
            "critical_model_results",
            "new_model_results",
            "archived_test_files",
            "batch_inference_results"
        ]
        
        # Exclude directories if specified
        exclude_dirs = []
        if args.exclude_dirs:
            exclude_dirs = [d.strip() for d in args.exclude_dirs.split(',')]
            
        dirs_to_scan = [d for d in default_dirs if d not in exclude_dirs]
    
    # Calculate cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=args.older_than)
    cutoff_timestamp = cutoff_date.timestamp()
    
    # Categories to process
    categories = args.categories.split(',') if args.categories != 'all' else ['all']
    
    json_files = []
    
    for directory in dirs_to_scan:
        directory_path = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory_path):
            logger.warning(f"Directory not found: {directory_path}")
            continue
            
        logger.info(f"Scanning for JSON files in: {directory}")
        
        # Find all JSON files in the directory and subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    file_stat = os.stat(file_path)
                    
                    # Check if the file is older than the cutoff date
                    if file_stat.st_mtime < cutoff_timestamp:
                        # Determine the category
                        category = categorize_json_file(file_path)
                        
                        # Check if this category should be processed
                        if categories[0] == 'all' or category in categories:
                            json_files.append({
                                'path': file_path,
                                'category': category,
                                'size': file_stat.st_size,
                                'modified': datetime.datetime.fromtimestamp(file_stat.st_mtime)
                            })
    
    logger.info(f"Found {len(json_files)} JSON files older than {args.older_than} days")
    
    # Limit the number of files to process if needed
    if len(json_files) > args.max_files:
        logger.warning(f"Limiting to {args.max_files} files (out of {len(json_files)} found)")
        # Sort by modification time, oldest first
        json_files.sort(key=lambda x: x['modified'])
        json_files = json_files[:args.max_files]
    
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

def process_file(file_info, args):
    """Process a single JSON file"""
    file_path = file_info['path']
    category = file_info['category']
    
    logger.debug(f"Processing {category} file: {file_path}")
    
    # If simulating, just return success
    if args.action == 'simulate':
        logger.debug(f"Simulation: would process {file_path}")
        return True, file_path, "Simulated processing"
    
    try:
        # Run the converter script to migrate this file
        script_dir = os.path.join(os.getcwd(), "scripts")
        converter_script = os.path.join(script_dir, "benchmark_db_converter.py")
        
        if not os.path.exists(converter_script):
            logger.error(f"Converter script not found at {converter_script}")
            return False, file_path, "Converter script not found"
        
        # Build command to run the converter
        cmd = [
            sys.executable,
            converter_script,
            "--input-file", file_path,
            "--output-db", args.db,
            "--categories", category
        ]
        
        if args.verbose:
            cmd.append("--verbose")
        
        # Import the converter module and call it directly
        sys.path.append(script_dir)
        from benchmark_db_converter import process_input_file
        
        # Connect to the database
        try:
            import duckdb
            conn = duckdb.connect(args.db)
            
            # Process the file
            success = process_input_file(conn, file_path)
            
            if success:
                # Commit changes
                conn.commit()
                
                # Take action on the file
                if args.action == 'archive':
                    archive_file(file_path, args.archive_dir)
                    return True, file_path, "Archived"
                elif args.action == 'remove':
                    os.remove(file_path)
                    return True, file_path, "Removed"
            else:
                return False, file_path, "Processing failed"
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False, file_path, str(e)
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return False, file_path, str(e)

def archive_file(file_path, archive_dir):
    """Archive a file by moving it to the archive directory"""
    # Create the archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)
    
    # Get the relative path within the repo
    try:
        rel_path = os.path.relpath(file_path, os.getcwd())
    except:
        # Fallback to just the basename if relpath fails
        rel_path = os.path.basename(file_path)
    
    # Create the archive path
    archive_path = os.path.join(archive_dir, rel_path)
    
    # Create parent directories if needed
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    
    # Move the file
    shutil.move(file_path, archive_path)
    logger.debug(f"Archived {file_path} to {archive_path}")
    
    return archive_path

def main():
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Find JSON files to process
    json_files = find_json_files(args)
    
    if not json_files:
        logger.info("No JSON files found to process")
        return
    
    # Ask for confirmation if not explicitly confirmed
    if not args.confirm:
        action_desc = {
            'archive': 'archived to',
            'remove': 'permanently removed',
            'simulate': 'simulated (no changes)'
        }
        
        print(f"\nFound {len(json_files)} JSON files to process.")
        print(f"Files will be migrated to the database and {action_desc[args.action]}")
        if args.action == 'archive':
            print(f"Archive directory: {args.archive_dir}")
        
        # Print a sample of files
        if len(json_files) > 0:
            print("\nSample files:")
            for i, file_info in enumerate(json_files[:5]):
                print(f"  - {file_info['path']} ({file_info['category']})")
            
            if len(json_files) > 5:
                print(f"  - ... and {len(json_files) - 5} more files")
        
        confirm = input("\nProceed with migration? (y/n): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Migration cancelled")
            return
    
    # Process files
    success_count = 0
    error_count = 0
    
    if args.parallel > 1:
        # Process files in parallel
        logger.info(f"Processing {len(json_files)} files in parallel with {args.parallel} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all files for processing
            futures = [executor.submit(process_file, file_info, args) for file_info in json_files]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                success, file_path, message = future.result()
                
                if success:
                    success_count += 1
                    logger.info(f"Successfully processed: {file_path}")
                else:
                    error_count += 1
                    logger.error(f"Failed to process: {file_path} - {message}")
                    
                    if not args.skip_errors:
                        logger.error("Stopping due to error. Use --skip-errors to continue on errors.")
                        break
    else:
        # Process files sequentially
        logger.info(f"Processing {len(json_files)} files sequentially")
        
        for file_info in json_files:
            success, file_path, message = process_file(file_info, args)
            
            if success:
                success_count += 1
                logger.info(f"Successfully processed: {file_path}")
            else:
                error_count += 1
                logger.error(f"Failed to process: {file_path} - {message}")
                
                if not args.skip_errors:
                    logger.error("Stopping due to error. Use --skip-errors to continue on errors.")
                    break
    
    # Print summary
    logger.info(f"\nMigration complete:")
    logger.info(f"  - Total files: {len(json_files)}")
    logger.info(f"  - Successfully processed: {success_count}")
    logger.info(f"  - Errors: {error_count}")
    
    if args.action == 'archive':
        logger.info(f"  - Files archived to: {args.archive_dir}")
    elif args.action == 'remove':
        logger.info(f"  - Files removed: {success_count}")
    elif args.action == 'simulate':
        logger.info(f"  - Simulation completed, no files were modified")

if __name__ == "__main__":
    main()