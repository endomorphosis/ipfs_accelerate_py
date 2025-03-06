#!/usr/bin/env python3
"""
IPFS Accelerate Legacy Test Results Migration Tool

This tool migrates legacy JSON test results for the IPFS Accelerate package to the DuckDB database.
It provides comprehensive validation, deduplication, and reporting capabilities.

Features:
- Searches for and identifies IPFS-related test result files
- Validates JSON files for correct format and data integrity
- Migrates results to structured DuckDB database
- Archives original JSON files after successful migration
- Generates detailed migration reports
- Support for different result formats and structures

Usage:
    python migrate_ipfs_test_results.py --input-dirs ./test_results ./archived_results
    python migrate_ipfs_test_results.py --input-dirs ./test_results --archive --report
    python migrate_ipfs_test_results.py --input-dirs ./test_results --delete --validate-strict
"""

import os
import sys
import json
import glob
import time
import hashlib
import argparse
import tarfile
import logging
import traceback
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ipfs_migration.log")
    ]
)
logger = logging.getLogger("ipfs_migration")

# Try to import DuckDB
try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False
    logger.error("DuckDB not installed. Please install with: pip install duckdb pandas")
    logger.error("Migration will not be performed.")

# Add parent directory to sys.path for proper imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from test_ipfs_accelerate import TestResultsDBHandler
    HAS_TEST_HANDLER = True
except ImportError:
    HAS_TEST_HANDLER = False
    logger.error("Could not import TestResultsDBHandler from test_ipfs_accelerate.py")
    logger.error("Make sure test_ipfs_accelerate.py is in the correct path.")


class IPFSTestResultSchema:
    """Schema definitions for IPFS test results validation."""
    
    # Core test result fields
    CORE_FIELDS = {
        "test_name": str,
        "status": str,
        "timestamp": (str, int, float),
        "execution_time": (float, int),
    }
    
    # IPFS-specific test fields
    IPFS_FIELDS = {
        "cid": str,
        "add_time": (float, int),
        "get_time": (float, int),
        "file_size": (int, float),
        "checkpoint_loading_time": (float, int),
        "dispatch_time": (float, int),
    }
    
    # Performance metric fields
    PERFORMANCE_FIELDS = {
        "throughput": (float, int),
        "latency": (float, int),
        "memory_usage": (float, int),
        "batch_size": int,
    }
    
    # Container operation fields
    CONTAINER_FIELDS = {
        "container_name": str,
        "image": str,
        "start_time": (float, int),
        "stop_time": (float, int),
        "operation": str,
    }
    
    # Configuration test fields
    CONFIG_FIELDS = {
        "config_section": str,
        "config_key": str,
        "expected_value": (str, int, float, bool),
        "actual_value": (str, int, float, bool),
    }
    
    @classmethod
    def validate_test_result(cls, result: Dict, strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate a test result against the schema.
        
        Args:
            result: The test result dictionary to validate
            strict: If True, requires core fields to be present
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check required fields if strict mode
        if strict:
            for field, field_type in cls.CORE_FIELDS.items():
                if field not in result:
                    errors.append(f"Missing required field: {field}")
                else:
                    # Check field type
                    if not isinstance(result[field], field_type):
                        if isinstance(field_type, tuple):
                            # Check against multiple allowed types
                            if not any(isinstance(result[field], t) for t in field_type):
                                errors.append(f"Field {field} has wrong type: {type(result[field]).__name__}, expected {field_type}")
                        else:
                            errors.append(f"Field {field} has wrong type: {type(result[field]).__name__}, expected {field_type.__name__}")
        
        # Check recognized fields and types (non-strict)
        for field, value in result.items():
            # Skip null values
            if value is None:
                continue
                
            # Check field against all schema categories
            schema_categories = [
                cls.CORE_FIELDS,
                cls.IPFS_FIELDS, 
                cls.PERFORMANCE_FIELDS,
                cls.CONTAINER_FIELDS,
                cls.CONFIG_FIELDS
            ]
            
            field_found = False
            for schema in schema_categories:
                if field in schema:
                    field_found = True
                    field_type = schema[field]
                    
                    # Validate type
                    if not isinstance(value, field_type):
                        if isinstance(field_type, tuple):
                            # Check against multiple allowed types
                            if not any(isinstance(value, t) for t in field_type):
                                errors.append(f"Field {field} has wrong type: {type(value).__name__}, expected one of {field_type}")
                        else:
                            errors.append(f"Field {field} has wrong type: {type(value).__name__}, expected {field_type.__name__}")
                    
                    break
            
            # Warning for unknown fields (not an error)
            if not field_found and field not in ["details", "error", "metadata", "notes"]:
                logger.debug(f"Unknown field in test result: {field}")
        
        return len(errors) == 0, errors


class IPFSResultMigrationTool:
    """
    Tool for migrating legacy IPFS test results from JSON files to DuckDB database.
    """
    
    def __init__(self, db_path: str = None, archive_dir: str = None):
        """
        Initialize the migration tool.
        
        Args:
            db_path: Path to the DuckDB database
            archive_dir: Directory for archiving JSON files
        """
        if not HAVE_DUCKDB or not HAS_TEST_HANDLER:
            logger.error("Required dependencies not available. Migration cannot proceed.")
            sys.exit(1)
            
        # Set database path
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        # Set archive directory
        self.archive_dir = archive_dir or "./archived_ipfs_results"
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Create database handler
        self.db_handler = TestResultsDBHandler(db_path=self.db_path)
        
        # Statistics for reporting
        self.stats = {
            "start_time": datetime.now(),
            "end_time": None,
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "skipped_files": 0,
            "total_results": 0,
            "valid_results": 0,
            "invalid_results": 0,
            "migrated_results": 0,
            "duplicate_results": 0,
            "errors": [],
            "processed_files": [],
            "archived_files": [],
            "deleted_files": []
        }
        
        # Set for tracking hashes of processed results
        self.processed_hashes = set()
    
    def find_ipfs_test_files(self, input_dirs: List[str], pattern: str = "*.json") -> List[str]:
        """
        Find IPFS test result files in the specified directories.
        
        Args:
            input_dirs: List of directories to search
            pattern: File pattern to search for (default: *.json)
            
        Returns:
            List of file paths
        """
        ipfs_files = []
        
        for directory in input_dirs:
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                continue
                
            # Use glob for file matching
            search_path = os.path.join(directory, pattern)
            files = glob.glob(search_path, recursive=True)
            
            # Look for nested files with double-star pattern
            if "**" not in pattern:
                nested_search = os.path.join(directory, "**", pattern)
                files.extend(glob.glob(nested_search, recursive=True))
            
            logger.info(f"Found {len(files)} JSON files in {directory}")
            
            # Filter for IPFS-related files
            for file_path in files:
                # Check if file appears to be IPFS-related
                # This heuristic could be improved
                file_name = os.path.basename(file_path)
                if ("ipfs" in file_name.lower() or 
                    "benchmark" in file_name.lower() or
                    "test_result" in file_name.lower()):
                    ipfs_files.append(file_path)
            
            logger.info(f"Identified {len(ipfs_files)} potential IPFS test files in {directory}")
                
        # Sort by modification time (newest first)
        ipfs_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        self.stats["total_files"] = len(ipfs_files)
        return ipfs_files
    
    def validate_json_file(self, file_path: str) -> Tuple[bool, Any]:
        """
        Validate a JSON file to ensure it contains valid IPFS test results.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Tuple of (is_valid, data)
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic validation - ensure it's a dict or list
            if not isinstance(data, (dict, list)):
                logger.warning(f"File is not a JSON object or array: {file_path}")
                return False, None
                
            # If it's an empty list, it's not valid
            if isinstance(data, list) and not data:
                logger.warning(f"File contains an empty array: {file_path}")
                return False, None
                
            # Look for IPFS-related keys in the data
            ipfs_related = False
            
            if isinstance(data, dict):
                # Look for IPFS keys
                ipfs_keys = ["ipfs_accelerate", "cid", "add_file", "get_file", 
                           "checkpoint", "dispatch", "ipfs_operations"]
                
                # Look for test result keys
                test_keys = ["test_name", "status", "execution_time", "result", "error"]
                
                # Check if any IPFS or test keys exist
                has_ipfs_keys = any(key in data for key in ipfs_keys)
                has_test_keys = any(key in data for key in test_keys)
                
                ipfs_related = has_ipfs_keys or has_test_keys
                
                # Check nested dictionaries if needed
                if not ipfs_related and "results" in data and isinstance(data["results"], (dict, list)):
                    nested = data["results"]
                    if isinstance(nested, dict):
                        has_ipfs_keys = any(key in nested for key in ipfs_keys)
                        has_test_keys = any(key in nested for key in test_keys)
                        ipfs_related = has_ipfs_keys or has_test_keys
            
            elif isinstance(data, list):
                # Check the first few items
                sample_size = min(5, len(data))
                for i in range(sample_size):
                    item = data[i]
                    if not isinstance(item, dict):
                        continue
                        
                    # Look for IPFS keys
                    ipfs_keys = ["ipfs_accelerate", "cid", "add_file", "get_file", 
                               "checkpoint", "dispatch", "ipfs_operations"]
                    
                    # Look for test result keys
                    test_keys = ["test_name", "status", "execution_time", "result", "error"]
                    
                    # Check if any IPFS or test keys exist
                    has_ipfs_keys = any(key in item for key in ipfs_keys)
                    has_test_keys = any(key in item for key in test_keys)
                    
                    if has_ipfs_keys or has_test_keys:
                        ipfs_related = True
                        break
            
            if not ipfs_related:
                logger.info(f"File does not appear to contain IPFS test results: {file_path}")
                
            return ipfs_related, data
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON file: {file_path}")
            return False, None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            self.stats["errors"].append(f"File read error: {file_path} - {e}")
            return False, None
    
    def extract_test_results(self, data: Any) -> List[Dict]:
        """
        Extract test results from the data structure.
        
        Args:
            data: The loaded JSON data
            
        Returns:
            List of test result dictionaries
        """
        results = []
        
        try:
            if isinstance(data, dict):
                # Check if it's a single test result
                if "test_name" in data or "status" in data:
                    results.append(data)
                # Check for nested results
                elif "results" in data:
                    nested = data["results"]
                    if isinstance(nested, list):
                        results.extend(nested)
                    elif isinstance(nested, dict):
                        # Dictionary of results
                        for key, value in nested.items():
                            if isinstance(value, dict):
                                # Add the key as context
                                value["test_name"] = value.get("test_name", key)
                                results.append(value)
                            elif isinstance(value, list):
                                for item in value:
                                    if isinstance(item, dict):
                                        # Add the key as context
                                        item["test_name"] = item.get("test_name", key)
                                        results.append(item)
                # Try to extract from any other structure
                else:
                    for key, value in data.items():
                        if isinstance(value, dict) and ("status" in value or "test_name" in value):
                            # This looks like a test result
                            value["test_name"] = value.get("test_name", key)
                            results.append(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and ("status" in item or "test_name" in item):
                                    results.append(item)
            
            elif isinstance(data, list):
                # Process list items
                for item in data:
                    if isinstance(item, dict):
                        if "test_name" in item or "status" in item:
                            results.append(item)
        
        except Exception as e:
            logger.error(f"Error extracting test results: {e}")
            self.stats["errors"].append(f"Extraction error: {e}")
        
        return results
    
    def hash_result(self, result: Dict) -> str:
        """
        Create a hash of a test result to identify duplicates.
        
        Args:
            result: The test result dictionary
            
        Returns:
            Hash string
        """
        # Create a stable representation of the result
        # Use a combination of fields that would make the result unique
        test_name = result.get("test_name", "")
        status = result.get("status", "")
        execution_time = result.get("execution_time", "")
        timestamp = result.get("timestamp", "")
        
        # If we have unique identifiers, use them
        if test_name and (status or execution_time):
            hash_str = f"{test_name}:{status}:{execution_time}:{timestamp}"
            return hashlib.md5(hash_str.encode()).hexdigest()
        
        # Otherwise, hash the whole dictionary 
        # First, create a stable JSON representation by sorting keys
        result_str = json.dumps(result, sort_keys=True)
        return hashlib.md5(result_str.encode()).hexdigest()
    
    def process_file(self, file_path: str, strict: bool = False) -> bool:
        """
        Process a JSON file and migrate its contents to the database.
        
        Args:
            file_path: Path to the JSON file
            strict: Whether to use strict validation
            
        Returns:
            True if migration was successful, False otherwise
        """
        logger.info(f"Processing file: {file_path}")
        
        # Validate the file
        is_valid, data = self.validate_json_file(file_path)
        
        if not is_valid or data is None:
            self.stats["invalid_files"] += 1
            return False
        
        self.stats["valid_files"] += 1
        self.stats["processed_files"].append(file_path)
        
        # Extract test results
        test_results = self.extract_test_results(data)
        logger.info(f"Extracted {len(test_results)} test results from {file_path}")
        
        self.stats["total_results"] += len(test_results)
        
        # Track individual file stats
        file_stats = {
            "migrated": 0,
            "invalid": 0,
            "duplicate": 0
        }
        
        # Process each test result
        for result in test_results:
            # Validate the result
            is_valid, errors = IPFSTestResultSchema.validate_test_result(result, strict=strict)
            
            if not is_valid:
                logger.warning(f"Invalid test result: {errors}")
                self.stats["invalid_results"] += 1
                file_stats["invalid"] += 1
                continue
            
            self.stats["valid_results"] += 1
            
            # Check for duplicates
            result_hash = self.hash_result(result)
            
            if result_hash in self.processed_hashes:
                logger.debug(f"Duplicate test result found (hash: {result_hash})")
                self.stats["duplicate_results"] += 1
                file_stats["duplicate"] += 1
                continue
            
            # Store in database
            try:
                # Mark as processed
                self.processed_hashes.add(result_hash)
                
                # Check required fields and add defaults
                if "timestamp" not in result:
                    # Use file modification time as fallback
                    result["timestamp"] = os.path.getmtime(file_path)
                    
                # Add source file information
                if "source_file" not in result:
                    result["source_file"] = os.path.basename(file_path)
                
                # Store the result
                if self.db_handler.store_test_result(result):
                    self.stats["migrated_results"] += 1
                    file_stats["migrated"] += 1
                else:
                    logger.warning(f"Failed to store test result in database")
                    
            except Exception as e:
                logger.error(f"Error storing test result: {e}")
                self.stats["errors"].append(f"Database error: {str(e)}")
        
        logger.info(f"File {file_path} - Migrated: {file_stats['migrated']}, Invalid: {file_stats['invalid']}, Duplicate: {file_stats['duplicate']}")
        
        # Return success if at least one result was migrated
        return file_stats["migrated"] > 0
    
    def archive_file(self, file_path: str) -> bool:
        """
        Archive a JSON file after successful migration.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            True if archiving was successful, False otherwise
        """
        try:
            # Create archive file path
            file_name = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(self.archive_dir, f"{timestamp}_{file_name}")
            
            # Copy to archive
            shutil.copy2(file_path, archive_path)
            
            self.stats["archived_files"].append(file_path)
            logger.info(f"Archived file {file_path} to {archive_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error archiving file {file_path}: {e}")
            self.stats["errors"].append(f"Archive error: {file_path} - {str(e)}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a JSON file after successful migration and archiving.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            os.remove(file_path)
            self.stats["deleted_files"].append(file_path)
            logger.info(f"Deleted file {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            self.stats["errors"].append(f"Delete error: {file_path} - {str(e)}")
            return False
    
    def create_archive_package(self, output_path: str = None) -> str:
        """
        Create a compressed archive of all processed files.
        
        Args:
            output_path: Path for the archive package
            
        Returns:
            Path to the created archive
        """
        if not self.stats["processed_files"]:
            logger.warning("No files to archive")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.archive_dir, f"ipfs_results_{timestamp}.tar.gz")
        
        try:
            with tarfile.open(output_path, "w:gz") as tar:
                for file_path in self.stats["processed_files"]:
                    if os.path.exists(file_path):
                        # Add file to archive with relative path
                        arcname = os.path.basename(file_path)
                        tar.add(file_path, arcname=arcname)
            
            logger.info(f"Created archive package: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating archive package: {e}")
            self.stats["errors"].append(f"Archive package error: {str(e)}")
            return None
    
    def run_migration(self, input_dirs: List[str], archive: bool = False, 
                     delete: bool = False, strict: bool = False) -> Dict:
        """
        Run the migration process.
        
        Args:
            input_dirs: List of directories to search for JSON files
            archive: Whether to archive files after migration
            delete: Whether to delete files after archiving
            strict: Whether to use strict validation
            
        Returns:
            Migration statistics
        """
        logger.info(f"Starting IPFS test results migration")
        logger.info(f"Input directories: {input_dirs}")
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"Archive directory: {self.archive_dir}")
        logger.info(f"Archive files: {archive}")
        logger.info(f"Delete files: {delete}")
        logger.info(f"Strict validation: {strict}")
        
        # Find IPFS test files
        json_files = self.find_ipfs_test_files(input_dirs)
        
        if not json_files:
            logger.warning("No IPFS test result files found")
            return self.stats
        
        # Process each file
        for file_path in json_files:
            success = self.process_file(file_path, strict=strict)
            
            if success and archive:
                archived = self.archive_file(file_path)
                
                if archived and delete:
                    self.delete_file(file_path)
        
        # Update final statistics
        self.stats["end_time"] = datetime.now()
        self.stats["duration"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        logger.info(f"Migration completed in {self.stats['duration']:.2f} seconds")
        logger.info(f"Files processed: {len(self.stats['processed_files'])}/{self.stats['total_files']}")
        logger.info(f"Results migrated: {self.stats['migrated_results']}/{self.stats['total_results']}")
        
        return self.stats
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a migration report.
        
        Args:
            output_file: Path to write the report to
            
        Returns:
            Report content
        """
        # Calculate duration
        if self.stats["end_time"] is None:
            self.stats["end_time"] = datetime.now()
            
        duration = self.stats["end_time"] - self.stats["start_time"]
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Create the report
        report_lines = []
        
        # Header
        report_lines.append("# IPFS Test Results Migration Report")
        report_lines.append("")
        report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Duration:** {hours:02d}:{minutes:02d}:{seconds:02d}")
        report_lines.append(f"**Database:** {self.db_path}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append(f"- **Total Files Found:** {self.stats['total_files']}")
        report_lines.append(f"- **Valid Files:** {self.stats['valid_files']}")
        report_lines.append(f"- **Invalid Files:** {self.stats['invalid_files']}")
        report_lines.append(f"- **Files Processed:** {len(self.stats['processed_files'])}")
        report_lines.append(f"- **Files Archived:** {len(self.stats['archived_files'])}")
        report_lines.append(f"- **Files Deleted:** {len(self.stats['deleted_files'])}")
        report_lines.append("")
        report_lines.append(f"- **Total Results Found:** {self.stats['total_results']}")
        report_lines.append(f"- **Valid Results:** {self.stats['valid_results']}")
        report_lines.append(f"- **Invalid Results:** {self.stats['invalid_results']}")
        report_lines.append(f"- **Results Migrated:** {self.stats['migrated_results']}")
        report_lines.append(f"- **Duplicate Results:** {self.stats['duplicate_results']}")
        report_lines.append("")
        
        # Migration rate
        if self.stats['total_results'] > 0:
            migration_rate = (self.stats['migrated_results'] / self.stats['total_results']) * 100
            report_lines.append(f"- **Migration Success Rate:** {migration_rate:.1f}%")
            report_lines.append("")
        
        # Errors
        if self.stats["errors"]:
            report_lines.append("## Errors")
            report_lines.append("")
            
            # Show the first 10 errors
            for i, error in enumerate(self.stats["errors"][:10]):
                report_lines.append(f"{i+1}. {error}")
                
            # Indicate if there are more errors
            if len(self.stats["errors"]) > 10:
                report_lines.append(f"... and {len(self.stats['errors']) - 10} more errors")
                
            report_lines.append("")
        
        # Processed files
        if self.stats["processed_files"]:
            report_lines.append("## Processed Files")
            report_lines.append("")
            
            # Show the first 20 files
            for i, file_path in enumerate(self.stats["processed_files"][:20]):
                report_lines.append(f"{i+1}. {os.path.basename(file_path)}")
                
            # Indicate if there are more files
            if len(self.stats["processed_files"]) > 20:
                report_lines.append(f"... and {len(self.stats['processed_files']) - 20} more files")
                
            report_lines.append("")
        
        # Combine the report
        report_content = "\n".join(report_lines)
        
        # Write to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_content)
                logger.info(f"Report written to: {output_file}")
            except Exception as e:
                logger.error(f"Error writing report to {output_file}: {e}")
        
        return report_content


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Migrate IPFS test results from JSON files to DuckDB database")
    
    parser.add_argument(
        "--input-dirs", "-i", nargs="+", required=True,
        help="Directories to search for JSON files")
    
    parser.add_argument(
        "--db-path", "-d",
        help="Path to DuckDB database (default: use BENCHMARK_DB_PATH env var or ./benchmark_db.duckdb)")
    
    parser.add_argument(
        "--archive-dir", "-a", default="./archived_ipfs_results",
        help="Directory for archiving JSON files")
    
    parser.add_argument(
        "--archive", action="store_true",
        help="Archive JSON files after migration")
    
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete JSON files after archiving (implies --archive)")
    
    parser.add_argument(
        "--validate-strict", action="store_true",
        help="Use strict validation for test results")
    
    parser.add_argument(
        "--report", "-r", action="store_true",
        help="Generate migration report")
    
    parser.add_argument(
        "--report-file", default="ipfs_migration_report.md",
        help="Path for the migration report")
    
    parser.add_argument(
        "--create-archive-package", action="store_true",
        help="Create a compressed archive package of all processed files")
    
    parser.add_argument(
        "--archive-package-path",
        help="Path for the archive package (default: ./archived_ipfs_results/ipfs_results_TIMESTAMP.tar.gz)")
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # If delete is specified, always archive
    if args.delete:
        args.archive = True
    
    # Create and run the migration tool
    migrator = IPFSResultMigrationTool(
        db_path=args.db_path,
        archive_dir=args.archive_dir
    )
    
    # Run migration
    stats = migrator.run_migration(
        input_dirs=args.input_dirs,
        archive=args.archive,
        delete=args.delete,
        strict=args.validate_strict
    )
    
    # Generate report if requested
    if args.report:
        report_content = migrator.generate_report(output_file=args.report_file)
        print("\n" + report_content)
    
    # Create archive package if requested
    if args.create_archive_package:
        archive_path = migrator.create_archive_package(output_path=args.archive_package_path)
        if archive_path:
            logger.info(f"Archive package created: {archive_path}")
    
    # Return success if any results were migrated
    return stats["migrated_results"] > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)