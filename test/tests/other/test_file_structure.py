#!/usr/bin/env python3
"""
Test file structure for generators and duckdb_api packages.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_structure")

# Get parent directory path
parent_dir = os.path.dirname(os.getcwd())
logger.info(f"Parent directory: {}parent_dir}")

def test_directories():
    """Test that directories have the expected structure."""
    logger.info("\n=== Testing Directory Structure ===")
    
    # Define directories to check
    directories = []],,
    "generators",
    "generators/test_generators",
    "generators/models",
    "generators/templates",
    "duckdb_api",
    "duckdb_api/core",
    "duckdb_api/schema",
    "duckdb_api/utils",
    ]
    
    success_count = 0
    for directory in directories:
        full_path = os.path.join(parent_dir, directory)
        if os.path.isdir(full_path):
            logger.info(f"Directory exists: {}directory} ✅")
            success_count += 1
        else:
            logger.error(f"Directory missing: {}directory} ❌")
    
            logger.info(f"Directory structure: {}success_count}/{}len(directories)} correct ({}success_count/len(directories)*100:.1f}%)")
            return success_count == len(directories)

def test_files():
    """Test that key files exist in the expected locations."""
    logger.info("\n=== Testing Key Files ===")
    
    # Define key files to check
    key_files = []],,
    "generators/__init__.py",
    "generators/test_generators/merged_test_generator.py",
    "generators/test_generators/simple_test_generator.py",
    "generators/models/skill_hf_bert.py",
    "duckdb_api/__init__.py",
    "duckdb_api/core/benchmark_db_query.py",
    "duckdb_api/schema/check_database_schema.py",
    "duckdb_api/utils/simulation_analysis.py",
    ]
    
    success_count = 0
    for file_path in key_files:
        full_path = os.path.join(parent_dir, file_path)
        if os.path.isfile(full_path):
            logger.info(f"File exists: {}file_path} ✅")
            success_count += 1
        else:
            logger.error(f"File missing: {}file_path} ❌")
    
            logger.info(f"Key files: {}success_count}/{}len(key_files)} found ({}success_count/len(key_files)*100:.1f}%)")
            return success_count == len(key_files)

def test_migration_counts():
    """Check if the number of files matches expected migration count."""
    logger.info("\n=== Testing Migration Counts ===")
    
    # Define expected file counts
    expected_counts = {}:
        "generators": 216,  # Total expected generator files
        "duckdb_api": 83,  # Total expected duckdb_api files
        }
    
        results = {}}
    
    for directory, expected_count in expected_counts.items():
        full_path = os.path.join(parent_dir, directory)
        file_count = 0
        
        # Count all Python files in the directory recursively
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith(".py"):
                    file_count += 1
        
                    percentage = (file_count / expected_count) * 100
                    status = "✅" if file_count >= expected_count * 0.9 else "❌"
                    results[]],,directory] = (file_count, expected_count, percentage, status)
        :
            logger.info(f"{}directory}: Found {}file_count} Python files (Expected: {}expected_count}) - {}percentage:.1f}% {}status}")
    
    # Check if counts are within acceptable range (at least 90% of expected):
    for _, (file_count, expected_count, _, _) in results.items():
        if file_count < expected_count * 0.9:
        return False
    
            return True

def main():
    """Main entry point."""
    logger.info("Testing file structure for generators and duckdb_api packages")
    
    directories_success = test_directories()
    files_success = test_files()
    counts_success = test_migration_counts()
    
    # Print overall summary
    logger.info("\n=== Overall Structure Test Summary ===")
    logger.info(f"Directory Structure: {}'PASSED' if directories_success else 'FAILED'}"):
    logger.info(f"Key Files: {}'PASSED' if files_success else 'FAILED'}"):
        logger.info(f"Migration Counts: {}'PASSED' if counts_success else 'FAILED'}")
    :
    if directories_success and files_success and counts_success:
        logger.info("\n✅ All structure tests passed! The migration file structure looks correct.")
        return 0
    else:
        logger.error("\n❌ Some structure tests failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())