#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Benchmark Databases Utility

This script merges multiple DuckDB benchmark databases into a single database for cross-platform
analysis. It's designed to be used in CI/CD pipelines to aggregate results from different
mobile platforms (Android and iOS).

Usage:
    python merge_benchmark_databases.py --output OUTPUT_DB [--input INPUT_DB [INPUT_DB ...]]
    [--input-dir INPUT_DIR] [--pattern PATTERN] [--verbose]

Examples:
    # Merge specific databases
    python merge_benchmark_databases.py --output merged_results.duckdb 
        --input android_results.duckdb ios_results.duckdb

    # Merge all .duckdb files from a directory
    python merge_benchmark_databases.py --output merged_results.duckdb 
        --input-dir benchmark_results/ --pattern "*.duckdb"

    # Merge from both specific files and a directory
    python merge_benchmark_databases.py --output merged_results.duckdb 
        --input previous_results.duckdb --input-dir new_results/ --pattern "benchmark_*.duckdb"

Date: April 2025
"""

import os
import sys
import glob
import logging
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports
try:
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


class BenchmarkDatabaseMerger:
    """
    Merges multiple DuckDB benchmark databases into a single database.
    
    This class handles the merging of benchmark results from different sources,
    ensuring that schemas are compatible and data is properly combined.
    """
    
    def __init__(self, 
                 output_db: str,
                 input_dbs: Optional[List[str]] = None,
                 input_dir: Optional[str] = None,
                 pattern: str = "*.duckdb",
                 verbose: bool = False):
        """
        Initialize the benchmark database merger.
        
        Args:
            output_db: Path to output merged DuckDB database
            input_dbs: Optional list of input database paths
            input_dir: Optional directory to search for databases
            pattern: Glob pattern to match database files in input_dir
            verbose: Enable verbose logging
        """
        self.output_db = output_db
        self.input_dbs = input_dbs or []
        self.input_dir = input_dir
        self.pattern = pattern
        self.verbose = verbose
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize output database connection
        self.db_api = None
        
        # Track statistics
        self.stats = {
            "input_files": 0,
            "benchmark_runs": 0,
            "device_platforms": set(),
            "model_names": set(),
            "errors": 0
        }
    
    def find_input_files(self) -> List[str]:
        """
        Find all input database files to merge.
        
        Returns:
            List of database file paths
        """
        input_files = list(self.input_dbs)
        
        # Add files from input directory if specified
        if self.input_dir and os.path.isdir(self.input_dir):
            pattern_path = os.path.join(self.input_dir, self.pattern)
            dir_files = glob.glob(pattern_path)
            logger.info(f"Found {len(dir_files)} files matching pattern '{self.pattern}' in '{self.input_dir}'")
            input_files.extend(dir_files)
        
        # Remove duplicates while preserving order
        unique_files = []
        seen = set()
        for file_path in input_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        # Verify files exist
        valid_files = []
        for file_path in unique_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"Input file does not exist: {file_path}")
        
        self.stats["input_files"] = len(valid_files)
        logger.info(f"Total input files to merge: {len(valid_files)}")
        return valid_files
    
    def connect_to_output_db(self) -> bool:
        """
        Connect to the output database.
        
        Returns:
            Success status
        """
        try:
            # If output database exists, make a backup first
            if os.path.exists(self.output_db):
                backup_path = f"{self.output_db}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                logger.info(f"Output database exists, creating backup at {backup_path}")
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(self.output_db, backup_path)
            
            # Connect to output database
            self.db_api = BenchmarkDBAPI(self.output_db)
            logger.info(f"Connected to output database: {self.output_db}")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to output database: {e}")
            return False
    
    def merge_database(self, input_db_path: str) -> bool:
        """
        Merge a single database into the output database.
        
        Args:
            input_db_path: Path to input database
            
        Returns:
            Success status
        """
        if not self.db_api:
            logger.error("Output database connection not initialized")
            return False
        
        try:
            logger.info(f"Merging database: {input_db_path}")
            
            # Connect to input database
            input_db_api = BenchmarkDBAPI(input_db_path)
            
            # Get benchmark runs from input database
            benchmark_runs = input_db_api.get_all_benchmark_runs()
            logger.info(f"Found {len(benchmark_runs)} benchmark runs in {input_db_path}")
            
            # Track statistics
            self.stats["benchmark_runs"] += len(benchmark_runs)
            
            # Process each benchmark run
            for run in benchmark_runs:
                # Get run details
                run_id = run.get("id")
                device_info = run.get("device_info", {})
                platform = device_info.get("platform", "unknown")
                model_name = run.get("model_name", "unknown")
                
                # Update statistics
                self.stats["device_platforms"].add(platform)
                self.stats["model_names"].add(model_name)
                
                if self.verbose:
                    logger.debug(f"Processing run: {run_id} ({platform}, {model_name})")
                
                # Get benchmark configurations
                configurations = input_db_api.get_benchmark_configurations(run_id)
                
                # Get benchmark results
                results = input_db_api.get_benchmark_results(run_id)
                
                # Insert into output database
                self.db_api.insert_benchmark_run(run)
                
                for config in configurations:
                    self.db_api.insert_benchmark_configuration(config)
                
                for result in results:
                    self.db_api.insert_benchmark_result(result)
            
            logger.info(f"Successfully merged database: {input_db_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error merging database {input_db_path}: {e}")
            self.stats["errors"] += 1
            return False
    
    def merge_all_databases(self) -> bool:
        """
        Merge all input databases into the output database.
        
        Returns:
            Success status
        """
        # Find input files
        input_files = self.find_input_files()
        
        if not input_files:
            logger.error("No input files found")
            return False
        
        # Connect to output database
        if not self.connect_to_output_db():
            return False
        
        # Merge each database
        successful_merges = 0
        for input_db_path in input_files:
            if self.merge_database(input_db_path):
                successful_merges += 1
        
        logger.info(f"Merged {successful_merges}/{len(input_files)} databases")
        return successful_merges > 0
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the merge operation.
        
        Returns:
            Dictionary with summary information
        """
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "output_database": self.output_db,
            "input_files": self.stats["input_files"],
            "successful_merges": self.stats["input_files"] - self.stats["errors"],
            "errors": self.stats["errors"],
            "benchmark_runs": self.stats["benchmark_runs"],
            "device_platforms": list(self.stats["device_platforms"]),
            "model_names": list(self.stats["model_names"]),
        }
    
    def print_summary(self) -> None:
        """Print a summary of the merge operation."""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("BENCHMARK DATABASE MERGE SUMMARY")
        print("="*80)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Output Database: {summary['output_database']}")
        print(f"Input Files: {summary['input_files']}")
        print(f"Successful Merges: {summary['successful_merges']}")
        print(f"Errors: {summary['errors']}")
        print(f"Benchmark Runs: {summary['benchmark_runs']}")
        print(f"Device Platforms: {', '.join(summary['device_platforms'])}")
        print(f"Number of Models: {len(summary['model_names'])}")
        print("="*80)
    
    def run(self) -> int:
        """
        Run the complete database merge process.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        logger.info(f"Starting benchmark database merge to {self.output_db}")
        
        # Merge all databases
        if not self.merge_all_databases():
            logger.error("Merge operation failed")
            return 1
        
        # Print summary
        self.print_summary()
        
        # Return success if at least one database was merged successfully
        if self.stats["errors"] < self.stats["input_files"]:
            return 0
        else:
            return 1


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Merge Benchmark Databases Utility")
    
    parser.add_argument("--output", required=True, help="Path to output merged DuckDB database")
    parser.add_argument("--input", nargs="+", help="Paths to input DuckDB databases")
    parser.add_argument("--input-dir", help="Directory to search for databases")
    parser.add_argument("--pattern", default="*.duckdb", help="Glob pattern to match database files (default: *.duckdb)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_dir:
        parser.error("At least one of --input or --input-dir must be specified")
    
    try:
        # Run merge operation
        merger = BenchmarkDatabaseMerger(
            output_db=args.output,
            input_dbs=args.input,
            input_dir=args.input_dir,
            pattern=args.pattern,
            verbose=args.verbose
        )
        
        return merger.run()
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())