#!/usr/bin/env python
"""
Benchmark Database Updater for the IPFS Accelerate Python Framework.

This module updates the benchmark database with new test results, performing merges
with existing data when appropriate.

Usage:
    python benchmark_db_updater.py --input-file ./performance_results/new_benchmark.json
    python benchmark_db_updater.py --scan-dir ./archived_test_results --incremental
"""

import os
import sys
import json
import glob
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

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import benchmark DB converter for reuse
try:
    from benchmark_db_converter import BenchmarkDBConverter
except ImportError:
    print("Error: benchmark_db_converter.py not found in the current directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkDBUpdater:
    """
    Updates the benchmark database with new test results.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark database updater.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        self.converter = BenchmarkDBConverter(output_db=db_path, debug=debug)
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized BenchmarkDBUpdater with DB: {db_path}")
    
    def update_from_file(self, file_path: str, category: str = None) -> bool:
        """
        Update the database with data from a single file.
        
        Args:
            file_path: Path to the JSON file
            category: Data category (if known, otherwise auto-detected)
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            # Convert file to DataFrame with appropriate schema
            file_category, df = self.converter.convert_file(file_path, category)
            
            if df.empty:
                logger.warning(f"No valid data found in file: {file_path}")
                return False
            
            # Create a dictionary of DataFrames with the single DataFrame
            dataframes = {file_category: df}
            
            # Save to DuckDB
            success = self.converter.save_to_duckdb(dataframes)
            
            if success:
                logger.info(f"Successfully updated database with data from: {file_path}")
                return True
            else:
                logger.error(f"Failed to update database with data from: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating database from file {file_path}: {e}")
            return False
    
    def update_from_directory(self, directory: str, categories: List[str] = None, 
                             incremental: bool = False, file_pattern: str = "*.json") -> Dict[str, int]:
        """
        Update the database with data from all files in a directory.
        
        Args:
            directory: Path to the directory containing JSON files
            categories: List of categories to include (or None for all)
            incremental: If True, only process files newer than the last update
            file_pattern: File pattern to match
            
        Returns:
            Dictionary with counts of processed files by category
        """
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return {}
        
        # Find all matching files
        pattern = os.path.join(directory, "**", file_pattern)
        json_files = glob.glob(pattern, recursive=True)
        
        if not json_files:
            logger.warning(f"No matching files found in directory: {directory}")
            return {}
        
        logger.info(f"Found {len(json_files)} files matching pattern {file_pattern} in {directory}")
        
        # Get most recent update time from database
        last_update_time = None
        if incremental:
            last_update_time = self._get_last_update_time()
            logger.info(f"Incremental mode: processing files newer than {last_update_time}")
        
        # Process files
        processed_counts = {}
        total_processed = 0
        
        for file_path in json_files:
            # Skip files older than last update if in incremental mode
            if incremental and last_update_time:
                file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime <= last_update_time:
                    logger.debug(f"Skipping file (older than last update): {file_path}")
                    continue
            
            # Process the file
            result = self.update_from_file(file_path)
            
            if result:
                # Detect category from file name or content
                category = self._detect_category_from_path(file_path)
                
                if category not in processed_counts:
                    processed_counts[category] = 0
                    
                processed_counts[category] += 1
                total_processed += 1
                
                if total_processed % 10 == 0:
                    logger.info(f"Processed {total_processed} files so far...")
        
        logger.info(f"Successfully processed {total_processed} files from {directory}")
        for category, count in processed_counts.items():
            logger.info(f"  {category}: {count} files")
        
        return processed_counts
    
    def _get_last_update_time(self) -> Optional[datetime.datetime]:
        """
        Get the timestamp of the most recent update to the database.
        
        Returns:
            Datetime of the last update, or None if no updates found
        """
        try:
            conn = duckdb.connect(self.db_path)
            
            # Query for the most recent timestamp across all tables
            tables = conn.execute("SHOW TABLES").fetchall()
            
            if not tables:
                logger.warning("No tables found in database")
                return None
            
            latest_timestamp = None
            
            for table in tables:
                table_name = table[0]
                
                # Check if the table has a timestamp or created_at column
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                column_names = [col[0].lower() for col in columns]
                
                timestamp_col = None
                if 'timestamp' in column_names:
                    timestamp_col = 'timestamp'
                elif 'created_at' in column_names:
                    timestamp_col = 'created_at'
                
                if timestamp_col:
                    # Get the most recent timestamp from this table
                    result = conn.execute(f"SELECT MAX({timestamp_col}) FROM {table_name}").fetchone()
                    if result and result[0]:
                        table_max = result[0]
                        
                        if isinstance(table_max, str):
                            try:
                                table_max = datetime.datetime.fromisoformat(table_max)
                            except ValueError:
                                try:
                                    table_max = datetime.datetime.strptime(table_max, '%Y-%m-%d %H:%M:%S')
                                except ValueError:
                                    continue
                        
                        if latest_timestamp is None or table_max > latest_timestamp:
                            latest_timestamp = table_max
            
            conn.close()
            return latest_timestamp
            
        except Exception as e:
            logger.error(f"Error getting last update time: {e}")
            return None
    
    def _detect_category_from_path(self, file_path: str) -> str:
        """
        Detect the category from the file path.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Category string ('performance', 'hardware', or 'compatibility')
        """
        file_name = os.path.basename(file_path).lower()
        
        if "performance" in file_name:
            return "performance"
        elif "hardware" in file_name:
            return "hardware"
        elif "compatibility" in file_name:
            return "compatibility"
        elif "test_results" in file_name:
            return "integration"
        elif "benchmark" in file_name:
            return "performance"
        else:
            # Try to infer from directory name
            dir_name = os.path.basename(os.path.dirname(file_path)).lower()
            
            if "performance" in dir_name:
                return "performance"
            elif "hardware" in dir_name:
                return "hardware"
            elif "compatibility" in dir_name:
                return "compatibility"
            else:
                # Default to performance
                return "performance"
    
    def track_processed_files(self, processed_files: List[str], tracking_file: str = ".benchmark_db_processed.txt") -> bool:
        """
        Track processed files to support incremental updates.
        
        Args:
            processed_files: List of processed file paths
            tracking_file: Path to the tracking file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing processed files
            existing_files = set()
            if os.path.exists(tracking_file):
                with open(tracking_file, 'r') as f:
                    existing_files = set(line.strip() for line in f if line.strip())
            
            # Add new processed files
            new_files = set(processed_files) - existing_files
            
            # Save all files
            with open(tracking_file, 'w') as f:
                for file_path in sorted(existing_files.union(new_files)):
                    f.write(f"{file_path}\n")
                    
            logger.info(f"Updated tracking file with {len(new_files)} new processed files")
            return True
            
        except Exception as e:
            logger.error(f"Error updating tracking file: {e}")
            return False
    
    def get_processed_files(self, tracking_file: str = ".benchmark_db_processed.txt") -> List[str]:
        """
        Get the list of previously processed files.
        
        Args:
            tracking_file: Path to the tracking file
            
        Returns:
            List of processed file paths
        """
        if not os.path.exists(tracking_file):
            return []
            
        try:
            with open(tracking_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
                
        except Exception as e:
            logger.error(f"Error reading tracking file: {e}")
            return []
    
    def clean_processed_files(self, older_than_days: int = 30, 
                             tracking_file: str = ".benchmark_db_processed.txt") -> int:
        """
        Clean up the tracking file by removing entries older than specified days.
        
        Args:
            older_than_days: Remove files older than this many days
            tracking_file: Path to the tracking file
            
        Returns:
            Number of entries removed
        """
        if not os.path.exists(tracking_file):
            return 0
            
        try:
            # Get existing processed files
            existing_files = self.get_processed_files(tracking_file)
            
            # Calculate cutoff date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
            
            # Filter files based on modification time
            retained_files = []
            removed_count = 0
            
            for file_path in existing_files:
                if os.path.exists(file_path):
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mtime > cutoff_date:
                        retained_files.append(file_path)
                    else:
                        removed_count += 1
                else:
                    # File no longer exists, skip it
                    removed_count += 1
            
            # Save retained files
            with open(tracking_file, 'w') as f:
                for file_path in retained_files:
                    f.write(f"{file_path}\n")
                    
            logger.info(f"Cleaned up tracking file: removed {removed_count} entries older than {older_than_days} days")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning tracking file: {e}")
            return 0
    
    def update_from_auto_store_db(self, output_dir: str = None) -> bool:
        """
        Update the database by checking auto-store results from test runners.
        This looks for temporary JSON results files in output_dir.
        
        Args:
            output_dir: Directory where auto-store files are saved
            
        Returns:
            True if successful, False otherwise
        """
        # Use default directory if not specified
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.db_path), "auto_store")
        
        # Check if directory exists
        if not os.path.exists(output_dir):
            logger.warning(f"Auto-store directory not found: {output_dir}")
            return False
        
        # Find all JSON files in the directory
        pattern = os.path.join(output_dir, "*.json")
        json_files = glob.glob(pattern)
        
        if not json_files:
            logger.info(f"No auto-store files found in {output_dir}")
            return True
        
        logger.info(f"Found {len(json_files)} auto-store files in {output_dir}")
        
        # Process each file
        processed_files = []
        for file_path in json_files:
            result = self.update_from_file(file_path)
            
            if result:
                processed_files.append(file_path)
                
                # Optionally move or delete the file after processing
                # os.remove(file_path)
        
        logger.info(f"Processed {len(processed_files)} auto-store files")
        return True

def main():
    """Command-line interface for the benchmark database updater."""
    parser = argparse.ArgumentParser(description="Benchmark Database Updater")
    parser.add_argument("--db", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--input-file", 
                       help="Input JSON file to update from")
    parser.add_argument("--category", choices=["performance", "hardware", "compatibility", "integration"],
                       help="Data category (if known, otherwise auto-detected)")
    parser.add_argument("--scan-dir", 
                       help="Directory to scan for JSON files")
    parser.add_argument("--file-pattern", default="*.json",
                       help="File pattern to match when scanning directory")
    parser.add_argument("--incremental", action="store_true",
                       help="Only process files newer than the last update")
    parser.add_argument("--track-processed", action="store_true",
                       help="Track processed files for future incremental updates")
    parser.add_argument("--tracking-file", default=".benchmark_db_processed.txt",
                       help="File to track processed files")
    parser.add_argument("--clean-tracking", action="store_true",
                       help="Clean up the tracking file")
    parser.add_argument("--older-than", type=int, default=30,
                       help="Clean up tracking entries older than this many days")
    parser.add_argument("--auto-store", action="store_true",
                       help="Check for auto-store files from test runners")
    parser.add_argument("--auto-store-dir", 
                       help="Directory where auto-store files are saved")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Create updater
    updater = BenchmarkDBUpdater(db_path=args.db, debug=args.debug)
    
    # Perform requested actions
    if args.input_file:
        # Update from single file
        if updater.update_from_file(args.input_file, args.category):
            logger.info(f"Successfully updated database from file: {args.input_file}")
            
            if args.track_processed:
                updater.track_processed_files([args.input_file], args.tracking_file)
        else:
            logger.error(f"Failed to update database from file: {args.input_file}")
            
    elif args.scan_dir:
        # Update from directory scan
        processed_counts = updater.update_from_directory(
            args.scan_dir, 
            categories=args.category.split(',') if args.category else None,
            incremental=args.incremental,
            file_pattern=args.file_pattern
        )
        
        if processed_counts and args.track_processed:
            # Get full paths of processed files
            processed_files = []
            for file_pattern in glob.glob(os.path.join(args.scan_dir, "**", args.file_pattern), recursive=True):
                file_path = os.path.abspath(file_pattern)
                processed_files.append(file_path)
            
            updater.track_processed_files(processed_files, args.tracking_file)
            
    elif args.clean_tracking:
        # Clean up tracking file
        removed_count = updater.clean_processed_files(args.older_than, args.tracking_file)
        logger.info(f"Removed {removed_count} entries from tracking file")
        
    elif args.auto_store:
        # Check for auto-store files
        if updater.update_from_auto_store_db(args.auto_store_dir):
            logger.info("Successfully processed auto-store files")
        else:
            logger.error("Failed to process auto-store files")
            
    else:
        # No specific action requested, print help
        parser.print_help()

if __name__ == "__main__":
    main()