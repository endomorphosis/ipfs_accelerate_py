#!/usr/bin/env python
"""
Test file for the DuckDB API module.

This script checks for syntax errors in the DuckDB API module by importing
all the main modules. It doesn't perform any actual functionality testing.
"""

import os
import sys
import logging
import importlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all the main modules can be imported."""
    modules = [
        "duckdb_api.core.benchmark_db_api",
        "duckdb_api.migration.benchmark_db_converter",
        "duckdb_api.schema.create_benchmark_schema",
        "duckdb_api.utils.benchmark_db_maintenance",
        "duckdb_api.utils.run_incremental_benchmarks",
        "duckdb_api.utils.simulation_detection",
        "duckdb_api.visualization.benchmark_db_query",
        "duckdb_api.visualization.benchmark_visualizer"
    ]
    
    success = True
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name}")
        except Exception as e:
            logger.error(f"Error importing {module_name}: {e}")
            success = False
    
    return success

def test_module_dependencies():
    """
    Check if required dependencies are available.
    This test only checks for the presence of dependencies but doesn't fail if they're missing.
    """
    dependencies = [
        "duckdb",
        "pandas",
        "fastapi",
        "uvicorn",
        "matplotlib",
        "seaborn",
        "numpy",
        "pyarrow"
    ]
    
    available_deps = []
    missing_deps = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            available_deps.append(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.warning("Install using: pip install " + " ".join(missing_deps))
    else:
        logger.info("All dependencies are available")
    
    logger.info(f"Available dependencies: {', '.join(available_deps)}")
    
    # This always returns True as we're just checking availability, not requiring them
    return True

if __name__ == "__main__":
    import_success = test_imports()
    dependency_check = test_module_dependencies()
    
    if import_success:
        logger.info("All modules imported successfully")
        sys.exit(0)
    else:
        logger.error("Some modules failed to import")
        sys.exit(1)