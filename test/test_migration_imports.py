#!/usr/bin/env python3
"""
Test file import functionality from generators and duckdb_api packages.
"""

import os
import sys
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_imports")

# Add parent directory to path
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.info(f"Added {parent_dir} to Python path")

def test_generators_imports():
    """Test importing modules from generators package."""
    logger.info("=== Testing Generators Package Imports ===")
    
    # Define modules to test import
    generator_modules = []]],,,
    "generators.test_generators.merged_test_generator",
    "generators.test_generators.simple_test_generator",
    "generators.test_generators.qualified_test_generator",
    "generators.models.skill_hf_bert",
    ]
    
    success_count = 0
    for module_name in generator_modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"Importing {module_name}... ✅")
            success_count += 1
        except Exception as e:
            logger.error(f"\1{str(e)}\3")
    
            logger.info(f"Generator imports: {success_count}/{len(generator_modules)} successful ({success_count/len(generator_modules)*100:.1f}%)")
            return success_count == len(generator_modules)

def test_duckdb_api_imports(skip_duckdb=True):
    """Test importing modules from duckdb_api package."""
    logger.info("\n=== Testing DuckDB API Package Imports ===")
    
    try:
        # First try to check if the base module exists
        import data.duckdb
        logger.info("Base duckdb_api module exists"):
    except ImportError:
        logger.error("Base duckdb_api module does not exist!")
            return False
    except Exception as e:
        logger.error(f"\1{e}\3")
            return False
        
    # If we're here, the base module exists, now check for Python files directly
            duckdb_api_files = []]],,,
            "core/verify_database_integration_fixed.py",
            "core/benchmark_db_query.py",
            "schema/check_database_schema.py",
            "utils/simulation_analysis.py",
            "migration/migrate_all_json_files.py"
            ]
    
            success_count = 0
            total_count = len(duckdb_api_files)
    
    for file_path in duckdb_api_files:
        full_path = os.path.join(parent_dir, "duckdb_api", file_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            logger.info(f"File exists: duckdb_api/{file_path} ✅")
            success_count += 1
        else:
            logger.error(f"File missing: duckdb_api/{file_path} ❌")
    
            logger.info(f"DuckDB API files: {success_count}/{total_count} found ({success_count/total_count*100:.1f}%)")
            return success_count == total_count

def test_directory_structure():
    """Test that directories have the expected structure."""
    logger.info("\n=== Testing Directory Structure ===")
    
    # Define directories to check
    directories = []]],,,
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
            logger.info(f"Directory exists: {directory} ✅")
            success_count += 1
        else:
            logger.error(f"Directory missing: {directory} ❌")
    
            logger.info(f"Directory structure: {success_count}/{len(directories)} correct ({success_count/len(directories)*100:.1f}%)")
            return success_count == len(directories)

def main():
    """Main entry point."""
    logger.info("Testing migration imports for generators and duckdb_api packages")
    
    generators_success = test_generators_imports()
    duckdb_api_success = test_duckdb_api_imports(skip_duckdb=True)
    directory_success = test_directory_structure()
    
    # Print overall summary
    logger.info("\n=== Overall Import Test Summary ===")
    logger.info(f"\1{'PASSED' if generators_success else 'FAILED'}\3"):
    logger.info(f"\1{'PASSED' if duckdb_api_success else 'FAILED'}\3"):
        logger.info(f"\1{'PASSED' if directory_success else 'FAILED'}\3")
    :
    if generators_success and duckdb_api_success and directory_success:
        logger.info("\n✅ All tests passed successfully! The migration appears to be working correctly.")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())