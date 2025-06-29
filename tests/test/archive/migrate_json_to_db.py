#!/usr/bin/env python3

import os
import sys
import json
import glob
import time
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib

# Add parent directory to sys.path for proper imports
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Try to import DuckDB
try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False
    print("Error: DuckDB not installed. Please install DuckDB with: pip install duckdb pandas")
    sys.exit(1)

# Try to import required modules
try:
    from test_ipfs_accelerate import TestResultsDBHandler
    HAS_TEST_MODULES = True
except ImportError:
    HAS_TEST_MODULES = False
    print("Error: Could not import TestResultsDBHandler. Make sure test_ipfs_accelerate.py is in the path.")
    sys.exit(1)

class JSONToDBMigrator:
    """
    Tool for migrating JSON test results to DuckDB database.
    This handles:
    - Finding JSON test result files
    - Parsing and validating JSON data
    - Storing results in DuckDB using TestResultsDBHandler
    - Creating migration reports and statistics
    - Archiving JSON files after successful migration
    """
    
    def __init__(self, db_path: str = None, archive_dir: str = None):
        """
        Initialize the migrator.
        
        Args:
            db_path: Path to DuckDB database
            archive_dir: Directory for archiving JSON files after migration
        """
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        self.archive_dir = archive_dir or "./archived_json_files"
        self.db_handler = TestResultsDBHandler(db_path=self.db_path)
        
        # Create archive directory if it doesn't exist
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir)
            
        # Stats for tracking migration progress
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "total_results": 0,
            "duplicates_found": 0,
            "start_time": time.time(),
            "end_time": None,
            "errors": []
        }
        
        # Cache for keeping track of processed records to avoid duplicates
        self.processed_hashes = set()
        
    def find_json_files(self, directories: List[str]) -> List[str]:
        """
        Find all JSON files in the specified directories.
        
        Args:
            directories: List of directories to search for JSON files
            
        Returns:
            List of JSON file paths
        """
        json_files = []
        
        for directory in directories:
            if not os.path.exists(directory):
                print(f"Warning: Directory does not exist: {directory}")
                continue
                
            # Find all JSON files in directory
            for pattern in ["*.json", "**/*.json"]:
                files = glob.glob(os.path.join(directory, pattern), recursive=True)
                json_files.extend(files)
                
        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        self.stats["total_files"] = len(json_files)
        return json_files
        
    def validate_json_file(self, json_path: str) -> Tuple[bool, Dict]:
        """
        Validate a JSON file to ensure it contains valid test results.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Tuple of (is_valid, data)
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Basic validation - ensure it's a dict or list
            if not isinstance(data, (dict, list)):
                return False, {}
                
            # If it's a list, ensure it's not empty
            if isinstance(data, list) and not data:
                return False, {}
                
            # If it's a dict, check for known test result keys
            if isinstance(data, dict):
                # Look for test result keys
                test_result_keys = ["status", "model_name", "hardware_type", "test_type"]
                has_test_keys = any(key in data for key in test_result_keys)
                
                # Look for performance keys
                perf_keys = ["throughput", "latency", "memory_usage", "batch_size"]
                has_perf_keys = any(key in data for key in perf_keys)
                
                if not (has_test_keys or has_perf_keys):
                    return False, {}
            
            return True, data
        except Exception as e:
            print(f"Error validating JSON file {json_path}: {e}")
            return False, {}
            
    def hash_result(self, result: Dict) -> str:
        """
        Create a hash of a result to identify duplicates.
        
        Args:
            result: Result dictionary
            
        Returns:
            Hash string
        """
        # Create a stable representation of the result
        model_name = result.get("model_name", "")
        hardware_type = result.get("hardware_type", "")
        test_type = result.get("test_type", "")
        timestamp = result.get("timestamp", "")
        
        # If we have enough identifying information, create a hash
        if model_name and hardware_type:
            hash_str = f"{model_name}:{hardware_type}:{test_type}:{timestamp}"
            return hashlib.md5(hash_str.encode()).hexdigest()
        
        # Otherwise, hash the whole dict
        result_str = json.dumps(result, sort_keys=True)
        return hashlib.md5(result_str.encode()).hexdigest()
        
    def process_file(self, json_path: str) -> bool:
        """
        Process a single JSON file and migrate its contents to the database.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            True if migration was successful, False otherwise
        """
        self.stats["processed_files"] += 1
        
        # Validate the JSON file
        is_valid, data = self.validate_json_file(json_path)
        if not is_valid:
            print(f"Skipping invalid JSON file: {json_path}")
            self.stats["failed_migrations"] += 1
            self.stats["errors"].append(f"Invalid JSON format: {json_path}")
            return False
            
        try:
            results_migrated = 0
            
            # Handle different data structures
            if isinstance(data, list):
                # For lists, migrate each item
                for item in data:
                    if isinstance(item, dict):
                        # Check for duplicates
                        result_hash = self.hash_result(item)
                        if result_hash in self.processed_hashes:
                            self.stats["duplicates_found"] += 1
                            continue
                            
                        # Store in database
                        if self.db_handler.store_test_result(item):
                            results_migrated += 1
                            self.processed_hashes.add(result_hash)
            elif isinstance(data, dict):
                # For dictionaries, check if it's a test result or a collection
                if "metadata" in data or "test_results" in data or "performance" in data:
                    # It's a test result container - extract test results
                    if "test_results" in data:
                        test_results = data["test_results"]
                        if isinstance(test_results, dict):
                            for model, model_results in test_results.items():
                                # For each model, process results
                                if isinstance(model_results, dict):
                                    for key, result in model_results.items():
                                        if isinstance(result, dict):
                                            # Add model name if not present
                                            if "model_name" not in result:
                                                result["model_name"] = model
                                                
                                            # Check for duplicates
                                            result_hash = self.hash_result(result)
                                            if result_hash in self.processed_hashes:
                                                self.stats["duplicates_found"] += 1
                                                continue
                                                
                                            # Store in database
                                            if self.db_handler.store_test_result(result):
                                                results_migrated += 1
                                                self.processed_hashes.add(result_hash)
                    
                    # Extract performance data if present
                    if "performance" in data and isinstance(data["performance"], dict):
                        for model, perf_data in data["performance"].items():
                            if isinstance(perf_data, dict):
                                # Add model name if not present
                                if "model_name" not in perf_data:
                                    perf_data["model_name"] = model
                                    
                                # Check for duplicates
                                perf_hash = self.hash_result(perf_data)
                                if perf_hash in self.processed_hashes:
                                    self.stats["duplicates_found"] += 1
                                    continue
                                    
                                # Store in database
                                if self.db_handler.store_test_result(perf_data):
                                    results_migrated += 1
                                    self.processed_hashes.add(perf_hash)
                else:
                    # It's a single test result
                    result_hash = self.hash_result(data)
                    if result_hash in self.processed_hashes:
                        self.stats["duplicates_found"] += 1
                    else:
                        # Store in database
                        if self.db_handler.store_test_result(data):
                            results_migrated += 1
                            self.processed_hashes.add(result_hash)
            
            self.stats["total_results"] += results_migrated
            
            if results_migrated > 0:
                self.stats["successful_migrations"] += 1
                return True
            else:
                self.stats["failed_migrations"] += 1
                self.stats["errors"].append(f"No results migrated from file: {json_path}")
                return False
        except Exception as e:
            print(f"Error processing file {json_path}: {e}")
            print(traceback.format_exc())
            self.stats["failed_migrations"] += 1
            self.stats["errors"].append(f"Error processing file {json_path}: {str(e)}")
            return False
            
    def archive_file(self, json_path: str) -> bool:
        """
        Archive a JSON file after successful migration.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            True if archiving was successful, False otherwise
        """
        try:
            # Create archive file path
            file_name = os.path.basename(json_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(self.archive_dir, f"{timestamp}_{file_name}")
            
            # Copy to archive
            import shutil
            shutil.copy2(json_path, archive_path)
            
            return True
        except Exception as e:
            print(f"Error archiving file {json_path}: {e}")
            return False
            
    def run_migration(self, directories: List[str], delete_after: bool = False) -> Dict:
        """
        Run the migration process for all JSON files in the specified directories.
        
        Args:
            directories: List of directories to search for JSON files
            delete_after: Whether to delete JSON files after successful migration
            
        Returns:
            Migration statistics
        """
        print(f"Starting migration from: {', '.join(directories)}")
        print(f"Database path: {self.db_path}")
        print(f"Archive directory: {self.archive_dir}")
        
        # Find all JSON files
        json_files = self.find_json_files(directories)
        print(f"Found {len(json_files)} JSON files")
        
        # Process each file
        for i, json_path in enumerate(json_files):
            print(f"Processing file {i+1}/{len(json_files)}: {json_path}")
            success = self.process_file(json_path)
            
            if success:
                # Archive the file
                archived = self.archive_file(json_path)
                
                # Delete if requested and archived successfully
                if delete_after and archived:
                    try:
                        os.remove(json_path)
                        print(f"Deleted file: {json_path}")
                    except Exception as e:
                        print(f"Error deleting file {json_path}: {e}")
        
        # Finish up and return stats
        self.stats["end_time"] = time.time()
        duration_seconds = self.stats["end_time"] - self.stats["start_time"]
        self.stats["duration_seconds"] = duration_seconds
        self.stats["duration_formatted"] = str(datetime.fromtimestamp(duration_seconds) - datetime.fromtimestamp(0))
        
        return self.stats
        
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a report of the migration process.
        
        Args:
            output_file: Path to write the report to
            
        Returns:
            Report text
        """
        duration_seconds = self.stats.get("duration_seconds", 0)
        hours, remainder = divmod(int(duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Create the report
        report = []
        report.append("# JSON to DuckDB Migration Report")
        report.append("")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Database:** {self.db_path}")
        report.append(f"**Archive Directory:** {self.archive_dir}")
        report.append("")
        report.append("## Summary")
        report.append("")
        report.append(f"- **Duration:** {hours:02d}:{minutes:02d}:{seconds:02d}")
        report.append(f"- **Total Files:** {self.stats['total_files']}")
        report.append(f"- **Processed Files:** {self.stats['processed_files']}")
        report.append(f"- **Successful Migrations:** {self.stats['successful_migrations']}")
        report.append(f"- **Failed Migrations:** {self.stats['failed_migrations']}")
        report.append(f"- **Total Results Migrated:** {self.stats['total_results']}")
        report.append(f"- **Duplicates Found:** {self.stats['duplicates_found']}")
        report.append("")
        
        # Add error summary if there are errors
        if self.stats["errors"]:
            report.append("## Errors")
            report.append("")
            for error in self.stats["errors"][:10]:  # Show only the first 10 errors
                report.append(f"- {error}")
                
            if len(self.stats["errors"]) > 10:
                report.append(f"- ... and {len(self.stats['errors']) - 10} more errors")
                
            report.append("")
        
        # Format as string
        report_text = "\n".join(report)
        
        # Write to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                print(f"Report written to: {output_file}")
            except Exception as e:
                print(f"Error writing report to {output_file}: {e}")
        
        return report_text

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Migrate JSON test results to DuckDB database")
    parser.add_argument("--directories", "-d", nargs="+", default=["./benchmark_results", "./archived_test_results", "./api_check_results"], 
                        help="Directories to search for JSON files")
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--archive-dir", default="./archived_json_files", help="Directory for archiving JSON files")
    parser.add_argument("--report", "-r", default="migration_report.md", help="Path to write the migration report")
    parser.add_argument("--delete", action="store_true", help="Delete JSON files after successful migration")
    args = parser.parse_args()
    
    # Create migrator
    migrator = JSONToDBMigrator(db_path=args.db_path, archive_dir=args.archive_dir)
    
    # Run migration
    stats = migrator.run_migration(args.directories, delete_after=args.delete)
    
    # Generate and print report
    report = migrator.generate_report(output_file=args.report)
    print("\n" + report)
    
    # Return success if there were any successful migrations
    return stats["successful_migrations"] > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)